import os
import numpy as np
import tqdm
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data.datareader import datareader
from model.seg_model import proposedNetwork
from model.lossesfunc import DiceBCELoss

# Veri okuma
dtreader = datareader(train_state=True)
valdtreader = datareader(train_state=False)

# Model oluşturma
model = proposedNetwork()

# Model derleme
model.compile(optimizer=Adam(lr=0.0001), loss=DiceBCELoss, metrics=['accuracy'])

# Model özeti
model.summary()

# Kontrol noktası yolu
ckpt_path = 'ckpt/Ubiris_dataset_batch2.h5'

# Var olan ağırlıkları yükleme
if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)
    print('checkpoint başarılı bir şekilde yüklendi.')

# Eğitim sırasında kullanılacak callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=0.00001)
earlystopper = EarlyStopping(patience=7, verbose=1)
checkpointer = ModelCheckpoint(ckpt_path, verbose=1, save_best_only=True)

# Model eğitimi
x_train_img= dtreader.images
x_train_mask = dtreader.masks
y_train = dtreader.masks
print('türü: ',type(x_train_img))
r = model.fit(x=x_train_img,  # Girişlerinizi  sağlayın
              y=y_train,  # Modelin beklendiği gibi hedefleri sağlayın
              validation_data=(valdtreader.images , valdtreader.masks),  # Doğrulama verilerini de düzenleyin
              callbacks=[earlystopper, reduce_lr, checkpointer],
              epochs=100,
              verbose=1,
              batch_size=2,
              shuffle=True)


# Modelin performansını değerlendirme
loss, accuracy = model.evaluate(valdtreader.images, valdtreader.masks, verbose=1)
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)
