import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

def load_face_cascade():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    return cascade

def detect_faces(face_cascade, gray_image):
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
    return faces

def crop_and_align_face(img_bgr, face_box, expand_ratio=0.25):
    x, y, w, h = face_box
    pad_w = int(w * expand_ratio)
    pad_h = int(h * expand_ratio)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_bgr.shape[1], x + w + pad_w)
    y2 = min(img_bgr.shape[0], y + h + pad_h)
    face_crop = img_bgr[y1:y2, x1:x2].copy()
    return face_crop

def compute_handcrafted_scores(img_bgr, face_box, face_crop):
    factors = []
    pose_score = 0
    try:
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=6)
        if len(eyes) >= 2:
            pose_score += 18
            factors.append("Cakeup")
        elif len(eyes) == 1:
            pose_score += 8
            factors.append("Satu mata terdeteksi")
    except Exception:
        pass
    fh, fw = face_crop.shape[:2]
    if fh > 0:
        ar = fw / fh
        if 0.85 < ar < 1.15:
            pose_score += 12
            factors.append("Golden Rasio")

    style_score = 0
    pad = 0
    try:
        crop_ext = face_crop.copy()
        hsv = cv2.cvtColor(crop_ext, cv2.COLOR_BGR2HSV)
        sat = np.mean(hsv[:,:,1])
        if sat > 110:
            style_score += 18
            factors.append("Your color is fine")
        elif sat > 60:
            style_score += 8
            factors.append("Your color is good")

        lap_var = cv2.Laplacian(cv2.cvtColor(crop_ext, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if lap_var > 150:
            style_score += 18
            factors.append("Buset")
        elif lap_var > 60:
            style_score += 8
            factors.append("Oke sih")
        else:
            factors.append("Burik")
    except Exception:
        lap_var = 0

    try:
        brightness = np.mean(face_crop)
        if 100 < brightness < 210:
            style_score += 12
            factors.append("Nice Composition")
        elif brightness <= 100:
            factors.append("U look so fine")
        else:
            factors.append("Damn u like a sun")
    except Exception:
        pass

    expression_score = 0
    try:
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        smiles = smile_cascade.detectMultiScale(gray_face, scaleFactor=1.6, minNeighbors=20)
        if len(smiles) > 0:
            expression_score += 25
            factors.append("Senyum Manis")
        else:
            h, w = gray_face.shape
            mouth_region = gray_face[int(0.6*h):h, int(0.2*w):int(0.8*w)]
            if mouth_region.size > 0 and np.std(mouth_region) > 10:
                expression_score += 12
                factors.append("Ekspresi mantep")
            else:
                expression_score += 4
                factors.append("Ekspresi lumayan")
    except Exception:
        factors.append("Ekspresi tidak terdeteksi")
    composition_score = 0
    try:
        h_img, w_img = img_bgr.shape[:2]
        x, y, w, h = face_box
        cx = x + w/2
        cy = y + h/2
        third_x = w_img / 3.0
        if abs(cx - third_x) < third_x/2 or abs(cx - 2*third_x) < third_x/2:
            composition_score += 10
            factors.append("Body dan wajah perfect")
        # background clutter: edges
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)
        if edge_density < 0.04:
            composition_score += 8
            factors.append("Background oke sih")
        else:
            factors.append("Background agak rame")
    except Exception:
        pass
    raw = (pose_score * 0.20 + style_score * 0.35 + expression_score * 0.25 + composition_score * 0.20)
    handcrafted_score = float(np.clip(raw, 0, 100))

    return handcrafted_score, factors, lap_var

def image_quality_checks(img, faces, lap_var_face):
    """Quality checks that penalize strongly for tiny faces, blur, extreme brightness"""
    img_area = img.shape[0] * img.shape[1]
    face_area = sum([w*h for (_,_,w,h) in faces])
    face_ratio = face_area / max(1, img_area)
    penalties = []
    penalty_score = 0
    if face_ratio < 0.02:
        penalties.append("Wajah terlalu kecil dalam foto")
        penalty_score += 20
    if lap_var_face < 30:
        penalties.append("Foto blur / kurang fokus")
        penalty_score += 18
    brightness = np.mean(img)
    if brightness < 50 or brightness > 240:
        penalties.append("Pencahayaan buruk")
        penalty_score += 15
    return penalty_score, penalties
