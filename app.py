import os
import cv2
import numpy as np
import base64
import random
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from utils import (
    load_face_cascade,
    detect_faces,
    crop_and_align_face,
    compute_handcrafted_scores,
    image_quality_checks
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

face_cascade = load_face_cascade()

MODEL_PATH = os.path.join("models", "fineshyt_model.h5")
trained_model = None
if os.path.exists(MODEL_PATH):
    try:
        trained_model = load_model(MODEL_PATH, compile=False)
        print("Loaded trained fineshyt model:", MODEL_PATH)
    except Exception as e:
        print("Failed to load trained model:", e)
else:
    print("No trained model found at", MODEL_PATH, "=> using heuristics + resnet proxy fallback")

MUSIC_LIBRARY = {
    'low': [
        {'title': 'Snooze', 'artist': 'SZA', 'spotify_id': '4iZ4pt7kvcaH6Yo8UoZ4s2', 'mood': 'RELAXED MODE'},
        {'title': 'vampire', 'artist': 'Olivia Rodrigo', 'spotify_id': '5VJETUjJ7y46bAz4min5IB', 'mood': 'DEEP VIBES'},
        {'title': 'Living Room Flow', 'artist': 'Jhené Aiko', 'spotify_id': '3nRU9i9BbhuD1an6Jx0c8r', 'mood': 'SMOOTH R&B'},
        {'title': 'Bathroom', 'artist': 'Montell Fish', 'spotify_id': '2ofebWbHf9ufZjeh6P2JwX', 'mood': 'INTIMATE VIBES'},
        {'title': 'All I Need', 'artist': 'Lloyd', 'spotify_id': '5lDtn3VZzi3A8cMquRZ9Xp', 'mood': 'SOFT ENERGY'}
    ],
    'medium': [
        {'title': 'Flowers', 'artist': 'Miley Cyrus', 'spotify_id': '0yLdNVWF3Srea0uzk55zFn', 'mood': 'BOSS VIBES'},
        {'title': 'Feel It', 'artist': 'Jacquees', 'spotify_id': '0c3ZZOZQe6K5x8woCjKs6D', 'mood': 'SEDUCTIVE FLOW'},
        {'title': 'Say It', 'artist': 'Tory Lanez', 'spotify_id': '6QgjcU0zLnzq5OrUoSZ3OK', 'mood': 'LATE NIGHT ENERGY'},
        {'title': 'Haunted', 'artist': 'Beyoncé', 'spotify_id': '3RWHTlVeqJskju2Hq5f4SG', 'mood': 'DARK R&B'},
        {'title': 'The Morning', 'artist': 'The Weeknd', 'spotify_id': '7yNK27ZTpHew0c55VvIJgm', 'mood': 'COOL CONFIDENCE'}
    ],
    'high': [
        {'title': 'Starboy', 'artist': 'The Weeknd', 'spotify_id': '7MXVkk9YMctZqd1Srtv4MB', 'mood': 'CONFIDENT MODE'},
        {'title': 'So Anxious', 'artist': 'Ginuwine', 'spotify_id': '5zA8vzDGqPl2AzZkEYQGKh', 'mood': 'CLASSIC ENERGY'},
        {'title': 'Me and Your Mama', 'artist': 'Childish Gambino', 'spotify_id': '1ZMiCix7XSAbfAJlEZWMCp', 'mood': 'POWER EMOTION'},
        {'title': 'In For It', 'artist': 'Tory Lanez', 'spotify_id': '7AXaQn1qdArj5elZFvWMeK', 'mood': 'DARK CONFIDENCE'}
    ]
}

def predict_with_model(face_img):
    if trained_model is None:
        return None
    try:
        img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224)).astype('float32')
        arr = np.expand_dims(img_resized, axis=0)
        arr = resnet_preprocess(arr)
        pred = trained_model.predict(arr, verbose=0)
        val = float(np.squeeze(pred))
        if val <= 1.001:
            return float(val * 100.0)
        return float(val)
    except Exception as e:
        print("model predict error:", e)
        return None

def combine_scores(model_score, handcrafted_score, penalties=0):
    if model_score is None:
        w_model = 0.0
        w_hand = 1.0
    else:
        w_model = 0.90
        w_hand = 0.30

    combined = (model_score or 0) * w_model + handcrafted_score * w_hand
    combined = combined - (penalties * 0.3)
    combined = float(np.clip(combined, 0, 100))
    if combined > 88:
        combined = min(100.0, combined + random.uniform(0, 5))
    return combined

def select_music(category):
    available = MUSIC_LIBRARY.get(category, [])
    if not available:
        return None
    return random.choice(available)

def calculate_fineshyt_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(face_cascade, gray)
    if len(faces) == 0:
        return 0.0, ["GA ADA MUKA NYA!"], "GA ADA MUKA NYA", "low"
    all_face_scores = []
    all_factors = []
    lap_var_face_min = 1e9

    for (x, y, w, h) in faces:
        face_crop = crop_and_align_face(img_bgr, (x, y, w, h))
        handcrafted_score, factors, lap_var = compute_handcrafted_scores(img_bgr, (x, y, w, h), face_crop)
        lap_var_face_min = min(lap_var_face_min, lap_var or 0)
        model_score = predict_with_model(face_crop)
        combined = combine_scores(model_score, handcrafted_score, penalties=0)
        all_face_scores.append(combined)
        all_factors.extend(factors)
    avg_score = float(sum(all_face_scores) / max(1, len(all_face_scores)))

    penalty_score, penalties = image_quality_checks(img_bgr, faces, lap_var_face_min)
    if penalty_score > 0:
        avg_score = max(0.0, avg_score - penalty_score)
        all_factors.extend(penalties)

    avg_score = float(np.clip(avg_score, 0.0, 100.0))

    if avg_score >= 80:
        category = "high"
        feedback = random.choice(["GOD DAYUM BRO!", "SICK BRO!", "ANGLE IS HERE MAN!"])
    elif avg_score >= 50:
        category = "medium"
        feedback = random.choice(["LUMAYAN SIH INI MAH.", "NOT BAD LAH."])
    else:
        category = "low"
        feedback = random.choice(["HELL NAH.", "KADAL JIR."])

    all_factors = list(dict.fromkeys([f for f in all_factors if f]))

    return avg_score, all_factors, feedback, category

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'GA ADA FOTO NYA LEK!'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'PILIH FOTO DULU BRO!'}), 400

    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'FORMAT FOTO GA BENER'}), 400

    score, factors, feedback, category = calculate_fineshyt_score(img)
    music = select_music(category)

    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode()

    return jsonify({
        'score': round(score, 1),
        'factors': factors,
        'feedback': feedback,
        'category': category,
        'music': music,
        'image': f"data:image/jpeg;base64,{img_str}"
    })

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    app.run(debug=True, host='0.0.0.0', port=5000)
