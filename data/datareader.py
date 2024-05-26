import os
import imageio
import numpy as np
from tqdm import tqdm
from skimage.transform import resize

class DataReader:
    def __init__(self, 
                 image_path="/content/dataset/images",
                 mask_path="/content/dataset/masks",
                 train_file="/content/dataset/UBIRIS_detection_train.txt",
                 val_file="/content/dataset/UBIRIS_detection_val.txt",
                 train_state=True):
        self.train_state = train_state
        self.image_path = image_path
        self.mask_path = mask_path
        #validation ve train işlemine göre txt'den dosyaları okur.
        self.file_path = train_file if train_state else val_file

        self.imagelist = self.get_image_list(self.file_path)
        self.num_images = len(self.imagelist)
        print('Number of images:', self.num_images)

        self.cur_index = 0
        self.images, self.masks = self.load_images_and_masks()

    def get_image_list(self, file_path):
        with open(file_path, 'r') as file:
            image_names = [line.strip() for line in file.readlines()]

        valid_images = []
        for name in image_names:
            if os.path.exists(os.path.join(self.image_path, f"{name}.jpg")):
                valid_images.append(name)
            else:
                print(f"Görsel bulunamadı: {os.path.join(self.image_path, name + '.jpg')}")
            
            if not os.path.exists(os.path.join(self.mask_path, f"{name}.png")):
                print(f"Maske bulunamadı: {os.path.join(self.mask_path, name + '.png')}")

        return valid_images

    def read_image_and_mask(self, name):
        image = imageio.imread(os.path.join(self.image_path, f"{name}.jpg"), mode='F', pilmode="RGB")
        mask = imageio.imread(os.path.join(self.mask_path, f"{name}.png"))

        mask[mask == 255] = 1  #maske piksel değerleri 0-1 olarak binary değere dönüşür.
        mask = np.expand_dims(mask, axis=2)  #maske height,weight,1 boyutuna genişletildi

        return image, mask

    def preprocess_image_and_mask(self, image, mask, height=256, width=256):
        # image ve mask'ı aynı boyuta yeniden boyutlandır
        image_resized = resize(image, (height, width))
        mask_resized = resize(mask, (height, width))

        return image_resized, mask_resized

    def load_images_and_masks(self):
        images = []
        masks = []
        for name in tqdm(self.imagelist):
            image, mask = self.read_image_and_mask(name)
            image_resized, mask_resized = self.preprocess_image_and_mask(image, mask)

            images.append(image_resized)
            masks.append(mask_resized)
            self.cur_index += 1

        images = np.array(images)
        masks = np.array(masks)

        return images, masks

# Kullanım
data_reader = DataReader()
images, masks = data_reader.images, data_reader.masks
