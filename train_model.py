import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

DATA_DIR = "dataset/images"
LABEL_CSV = "dataset/labels.csv"
OUT_MODEL = "models/fineshyt_model.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4

os.makedirs("models", exist_ok=True)

df = pd.read_csv(LABEL_CSV)
df = df[df['score'].notna()]
df = df[df['score'] >= 0]
if df.empty:
    raise SystemExit("Tidak ada data terlabel. Isi dataset/labels.csv dulu (score 0..100).")

df['score_norm'] = (df['score'].clip(0,100) / 100.0)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_frac = 0.8
train_count = int(len(df) * train_frac)
train_df = df.iloc[:train_count]
val_df = df.iloc[train_count:]

print(f"Total images: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)}")

AUTOTUNE = tf.data.AUTOTUNE

def preprocess_path_label(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, label

def df_to_dataset(df_in, shuffle=False, augment=False):
    paths = [os.path.join(DATA_DIR, f) for f in df_in['filename'].tolist()]
    labels = df_in['score_norm'].astype(np.float32).tolist()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(preprocess_path_label, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1024)
    if augment:
        def aug_fn(img, lbl):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.08)
            img = tf.image.random_contrast(img, 0.95, 1.05)
            return img, lbl
        ds = ds.map(aug_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = df_to_dataset(train_df, shuffle=True, augment=True)
val_ds = df_to_dataset(val_df, shuffle=False, augment=False)

base = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base.trainable = False

inp = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inp, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inp, out)
model.compile(optimizer=tf.keras.optimizers.Adam(LR),
              loss='mse',
              metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')])

model.summary()

callbacks = [
    ModelCheckpoint(OUT_MODEL, save_best_only=True, monitor='val_mae', mode='min', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_mae', patience=6, restore_best_weights=True, verbose=1)
]

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

base.trainable = True
fine_tune_at = len(base.layers) - 30
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='mse',
              metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')])

model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)
model.save(OUT_MODEL)
print("Saved model:", OUT_MODEL)
