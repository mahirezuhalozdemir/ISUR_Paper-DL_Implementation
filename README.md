# ISUR_Paper-DL_Implementation
This repository is an implementation of the [ISUR paper](https://ieeexplore.ieee.org/document/9721475) on image segmentation with UNet and Resnet

Bu makalede, gürültülü iris görüntülerinin segmentasyonu için ISUR adında bir yöntem önerilmektedir.
Model oluşturma süreci şu şekilde özetlenebilir:
-   **1. İris Tespiti:** İlk olarak, irisAttention adlı bir CNN detektörü kullanılarak iris bölgesinin koordinatları (sınırlayıcı kutu) elde edilir ve bir attention maskesi oluşturulur.
-   **2. İris Segmentasyonu:** Daha sonra, bu attention maskesi kullanılarak yeni bir CNN ile göz resmi segmente edilir. Bu aşamada, ResNet-18 ve SE bloğu gibi bileşenler kullanılır.
>   Model eğitimi için; Python (3.7), Keras (2.1.0) ve Tensorflow-gpu (1.14.0) kullanılarak kodlanmıştır.
Dataset:
*   CASIA-Iris version 4 distance verileri için: http://biometrics.idealtest.org/#/datasetDetail/4
    * 142 kişiden alınan 2,567 iris fotoğrafı içerir.
    * Veriler 2 sete bölünür. Her sette; eğitim için 296, değerlendirme için 99 fotoğraf içerir.
    * Fotoğraf boyutları: 480×640 pixel
*   UBIRIS verileri için: http://iris.di.ubi.pt/
    * 261 kişiden alınan 11,102 iris fotoğrafı içerir.
    * İki set için fotoğrafların 2250'si kullanıldı.
    * Eğitim için; 1,687 ve değerlendirme için 563 fotoğraf kullanıldı.
    * Fotoğraf boyutları: 300x400 pixel
