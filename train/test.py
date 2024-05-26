import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

from keras.models import load_model
from model.lossesfunc import DiceBCELoss,DiceLoss

# Modeli yükle
model = load_model('/content/ckpt/Ubiris_dataset_batch2.h5',custom_objects={'DiceBCELoss': DiceBCELoss,'DiceLoss':DiceLoss})


file_name = "C11_S1_I14_L"
img_dir = "/content/dataset/images"
mask_dir = "/content/dataset/masks"


# Tam dosya yolunu oluşturma
image_path = f"{img_dir}/{file_name}.jpg"

# image_path = f"/content/dataset/images/{file_name}.jpg"
mask_path = f"{mask_dir}/{file_name}.png"

# Görüntüyü yükle
image = imageio.imread(image_path)
# Gerçek maskeyi yükle
mask = imageio.imread(mask_path)
print('img shape: ', image.shape)
print('mask shape: ',mask.shape)

image = resize(image,(256,256))
mask = resize(mask,(256,256))

image_for_prediction = np.expand_dims(image, axis=0)

predicted_mask = model.predict(image_for_prediction)

# Görüntü, gerçek maske ve tahmin edilen maskeyi görselleştir
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Orijinal Görüntü')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('Gerçek Maske')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(predicted_mask[0], cmap='gray')
plt.title('Tahmin Edilen Maske')
plt.axis('off')

plt.savefig(f"{file_name}.png")
plt.show()
