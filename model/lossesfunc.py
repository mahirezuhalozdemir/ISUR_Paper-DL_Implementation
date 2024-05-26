from keras import backend as K
from keras.losses import categorical_crossentropy, binary_crossentropy

#modelin kayıp fonksiyonu olarak Dice Loss ve BCE kombinasyonu kullanılmıştır
def DiceLoss(y_true, y_pred, smooth=1e-15):

    # gerçek ve tahmini değerler
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)

    #dice_loss küme benzerliğini ölçer
    # dice coeff = (2xkesişim) / birleşim
    intersection = K.sum(y_true * y_pred)
    #smooth parametresi, 0 bölme hatalarını önlemek için eklenmiş küçük bir değerdir
    dice_loss = (2 * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    return 1 - dice_loss

def DiceBCELoss(y_true, y_pred):
    #segmentasyon için kullanılan Dice loss ile
    #sınıflandırma için kullanılan BCE fonksiyonları kullanılarak
    # segmentasyon performansı arttırılmıştır

    #dice loss -> segmentasyon doğruluğu
    #bce loss -> piksel sınıflandırma 
    loss = 0.7 * binary_crossentropy(y_true, y_pred)

    dice = DiceLoss(y_true, y_pred)

    loss += (0.3 * dice)
    return loss