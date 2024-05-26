import torch
import torch.nn as nn
import torch.nn.functional as F

class se_block(nn.Module):
    # kanal özellik değerlerine göre önem ağırlıklarının belirlendiği sınıftır
    # squeeze excitation blok yapısı
    def __init__(self, channels, reduction_ratio=16):
        super(se_block, self).__init__()
        # Ortalama havuzlama (sıkıştırma) işlemi
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # İlk tam bağlı (fully connected) katman, kanalları reduction_ratio ile azaltır
        self.fc1 = nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False)
        # ReLU aktivasyon fonksiyonu
        self.relu = nn.ReLU(inplace=True)
        # İkinci tam bağlı (fully connected) katman, kanalları eski boyutuna getirir
        self.fc2 = nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        # Sigmoid aktivasyon fonksiyonu, değerleri 0-1 aralığına çeker
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Giriş tensorünün boyutlarını alır (batch_size, channels, height, width)
        b, c, _, _ = x.size()
        # Ortalama havuzlama ve yeniden şekillendirme işlemi (sıkıştırma)
        y = self.avg_pool(x).view(b, c)
        # İlk tam bağlı katmanı uygular ve ReLU aktivasyonu
        y = self.relu(self.fc1(y))
        # İkinci tam bağlı katmanı uygular ve Sigmoid aktivasyonu
        y = self.sigmoid(self.fc2(y))
        # Yeniden şekillendir ve giriş ile çarp (kanal önem ağırlıklarını uygula)
        y = y.view(b, c, 1, 1)
        return x * y
