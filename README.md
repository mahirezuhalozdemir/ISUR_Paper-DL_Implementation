# ISUR Paper Implementation
This repository is an implementation of the [ISUR paper](https://ieeexplore.ieee.org/document/9721475) on image segmentation with UNet and Resnet
### [En](#about-the-project)/[Tr](#proje-hakkında)

# About The Project
This repo contains the implementation of the proposed ISUR method for segmentation of noisy iris images.
The model building process can be summarised as follows:
-   **1. Iris Detection:** First, the coordinates of the iris region (bounding box) are obtained using a CNN detector called irisAttention and an attention mask is generated.
-   **2. Iris Segmentation:** Then, the iris image is segmented with a new CNN using the attention mask. At this stage, components such as ResNet-18 and SE block are used.
  
  ```sh
For model training: Python (3.7), Keras (2.1.0) ve Tensorflow-gpu (1.14.0) were used.
```

>   **In this project, only the segmentation part of the model was applied.**
## Dataset:
*   For CASIA-Iris version 4 distance dataset: http://biometrics.idealtest.org/#/datasetDetail/4
    * Contains 2.567 iris photographs from 142 individuals.
    * The data is divided into 2 sets. Each set contains 296 photos for training and 99 photos for evaluation.
    * Image dimensions 480×640 pixels
*   For UBIRIS dataset: http://iris.di.ubi.pt/
    * Contains 11.102 iris photographs from 261 individuals.
    * 2250 of the photographs were used for the two sets.
    * 1,687 photographs were used for training and 563 for evaluation.
    * Image dimensions 300×400 pixels

### **Please see the report for detailed information.** [➡️](https://github.com/mahirezuhalozdemir/ISUR_Paper-DL_Implementation/blob/main/isur_implementation_report.pdf)
## Results
<p align="center"> <img width="500"  src="https://github.com/mahirezuhalozdemir/ISUR_Paper-DL_Implementation/blob/main/img/mask_1.png?raw=true"></p>
<p align="center"> <img width="500"  src="https://github.com/mahirezuhalozdemir/ISUR_Paper-DL_Implementation/blob/main/img/mask_2.png?raw=true"></p>
<p align="center"> <img width="500"  src="https://github.com/mahirezuhalozdemir/ISUR_Paper-DL_Implementation/blob/main/img/mask_3.png?raw=true"></p>

# Proje Hakkında
Bu repo, gürültülü iris görüntülerinin segmentasyonu için önerilen ISUR yönteminin implementasyonunu içerir.
Model oluşturma süreci şu şekilde özetlenebilir:
-   **1. İris Tespiti:** İlk olarak, irisAttention adlı bir CNN detektörü kullanılarak iris bölgesinin koordinatları (sınırlayıcı kutu) elde edilir ve bir attention maskesi oluşturulur.
-   **2. İris Segmentasyonu:** Daha sonra, attention maskesi kullanılarak yeni bir CNN ile iris görüntüsü segmente edilir. Bu aşamada, ResNet-18 ve SE bloğu gibi bileşenler kullanılır.

  ```sh
Model eğitimi için: Python (3.7), Keras (2.1.0) ve Tensorflow-gpu (1.14.0) kullanılarak kodlanmıştır.
```
>   **Bu çalışmada sadece modelin segmentasyon kısmı uygulanmıştır.**

## Veriseti:
*   CASIA-Iris version 4 distance verileri için: http://biometrics.idealtest.org/#/datasetDetail/4
    * 142 kişiden alınan 2,567 iris fotoğrafı içerir.
    * Veriler 2 sete bölünür. Her sette; eğitim için 296, değerlendirme için 99 fotoğraf içerir.
    * Fotoğraf boyutları: 480×640 pixel
*   UBIRIS verileri için: http://iris.di.ubi.pt/
    * 261 kişiden alınan 11,102 iris fotoğrafı içerir.
    * İki set için fotoğrafların 2250'si kullanıldı.
    * Eğitim için; 1,687 ve değerlendirme için 563 fotoğraf kullanıldı.
    * Fotoğraf boyutları: 300x400 pixel
### **Detaylı bilgi için raporu inceleyiniz.** [➡️](https://github.com/mahirezuhalozdemir/ISUR_Paper-DL_Implementation/blob/main/isur_implementation_report.pdf)

## Sonuçlar
<p align="center"> <img width="500"  src="https://github.com/mahirezuhalozdemir/ISUR_Paper-DL_Implementation/blob/main/img/mask_1.png?raw=true"></p>
<p align="center"> <img width="500"  src="https://github.com/mahirezuhalozdemir/ISUR_Paper-DL_Implementation/blob/main/img/mask_2.png?raw=true"></p>
<p align="center"> <img width="500"  src="https://github.com/mahirezuhalozdemir/ISUR_Paper-DL_Implementation/blob/main/img/mask_3.png?raw=true"></p>



