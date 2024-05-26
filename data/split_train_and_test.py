import os
import random

# Dosya yolları
image_dir = '/content/dataset/images'
train_txt_path = '/content/dataset/UBIRIS_detection_train.txt'
val_txt_path = '/content/dataset/UBIRIS_detection_val.txt'

# Tüm resim dosyalarının isimlerini al
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Dosyaları karıştır
random.shuffle(image_files)

# %80 - %20 oranında böl
split_index = int(len(image_files) * 0.8)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# Eğitim dosyalarının isimlerini kaydet
with open(train_txt_path, 'w') as train_file:
    for file_name in train_files:
        train_file.write(os.path.splitext(file_name)[0] + '\n')  # Uzantıyı kaldır ve yeni satıra yaz

# Doğrulama dosyalarının isimlerini kaydet
with open(val_txt_path, 'w') as val_file:
    for file_name in val_files:
        val_file.write(os.path.splitext(file_name)[0] + '\n')  # Uzantıyı kaldır ve yeni satıra yaz

print(f"Eğitim dosyaları ({len(train_files)} adet) '{train_txt_path}' dosyasına kaydedildi.")
print(f"Doğrulama dosyaları ({len(val_files)} adet) '{val_txt_path}' dosyasına kaydedildi.")
