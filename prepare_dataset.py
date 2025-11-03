import os
import pandas as pd

dataset_dir = "dataset/images"
labels_path = "dataset/labels.csv"

os.makedirs(dataset_dir, exist_ok=True)

image_files = sorted([
    f for f in os.listdir(dataset_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if not image_files:
    raise SystemExit("❌ Folder dataset/images kosong, masukkan gambar dulu!")
if not os.path.exists(labels_path):
    df = pd.DataFrame({
        "filename": image_files,
        "score": [-1] * len(image_files)
    })
    df.to_csv(labels_path, index=False)
    print("✅ File labels.csv baru dibuat otomatis.")
else:
    old_labels = pd.read_csv(labels_path)
    label_dict = dict(zip(old_labels["filename"], old_labels["score"]))
    new_entries = 0
    for img in image_files:
        if img not in label_dict:
            label_dict[img] = -1
            new_entries += 1
    removed = [f for f in old_labels["filename"] if f not in image_files]
    if removed:
        print(f"⚠️ Menghapus {len(removed)} data lama yang ga ada di folder images.")
        for r in removed:
            label_dict.pop(r, None)
    updated_df = pd.DataFrame({
        "filename": list(label_dict.keys()),
        "score": list(label_dict.values())
    })
    updated_df.to_csv(labels_path, index=False)
    
print(f"✅ File labels.csv diperbarui ({new_entries} file baru ditambahkan).")
print(f"\nTotal gambar: {len(image_files)}")
print(f"Total data di CSV: {len(pd.read_csv(labels_path))}")
print("Dataset siap digunakan 🚀")
