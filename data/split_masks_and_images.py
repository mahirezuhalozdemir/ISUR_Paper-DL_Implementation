import os
import shutil
import zipfile

# Zip Dosya yolu
extracted_folder = '/content/dataset_/ubipr'

# PNG ve JPG dosyalarını kaydedeceğimiz klasörleri oluşturun
png_dir = '/content/dataset/masks'
jpg_dir = '/content/dataset/images'

os.makedirs(png_dir, exist_ok=True) #klasör zaten varsa exist_ok=true sayesinde hata vermez
os.makedirs(jpg_dir, exist_ok=True)

# .zip dosyasını açın ve dosyaları uygun klasörlere ayırın
for root, dirs, files in os.walk(extracted_folder):
    for file_name in files:
        file_extension = os.path.splitext(file_name)[1].lower()
        if file_extension == '.png':
            # PNG dosyaları maskelerdir
            # PNG dosyalarını masks klasörüne taşı
            shutil.move(os.path.join(root, file_name), png_dir)
        elif file_extension == '.jpg':
            # JPG dosyaları fotoğraflardır
            # JPG dosyalarını images klasörüne taşı
            shutil.move(os.path.join(root, file_name), jpg_dir)

print("PNG ve JPG dosyaları başarıyla ayrıldı ve uygun klasörlere taşındı.")
