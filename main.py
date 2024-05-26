#modelini colab ile eğitmek için;
#1-) ubipr datasetini drive'a taşıyın
#2-) yeni bir notebook oluşturun
#3-) proje kodlarını drive content içerisine ekleyin
#4-) aşağıdaki kodları sırayla colab notebook'da çalıştırın

#Drive bağlantısı için
from google.colab import drive
drive.mount('/content/drive')


#Dataset zip'den çıkarmak için
!unzip /content/drive/My\ Drive/ubipr.zip -d /content/dataset_


#Fotoğrafları ve maskeleri ayırmak için
!python3 split_masks_and_images.py



#Görüntü sayısı kontrolü için 
import os
def count_items_in_directory(directory_path,directory_path2):
    # Tüm ögeleri listele
    items = os.listdir(directory_path)
    items2 = os.listdir(directory_path2)
    # Öge sayısını döndür
    return len(items), len(items2)

#klasör yolu
directory_path = '/content/dataset/images'
directory_path2 = '/content/dataset/masks'

# Klasördeki öge sayısını öğrenin
number_of_items,number_of_items2 = count_items_in_directory(directory_path,directory_path2)

print(f'{directory_path} klasöründe {number_of_items} öge bulunmaktadır.')
print(f'{directory_path2} klasöründe {number_of_items2} öge bulunmaktadır.')



#Train ve Validation Verilerini Ayırmak İçin
!python3 split_train_and_test.py



#Eğitim için
!python3 train.py




#Test İçin
!python3 test.py