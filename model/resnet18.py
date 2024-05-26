import torch
import torch.nn as nn
import torch.nn.functional as F
from squeezeexcitation import se_block

#bu dosya model içerisinde kullanılan Resnet18 mimarisini oluşturur
#Atlamalı bağlantılar olduğu için iki tür katman bulunur: identity ve conv katmanları

def identity_block(input_tensor, kernel_size, filters, stage, block):
    #identity block; atlama katmanlarıdır
    #iki çıktıyı birbirine ekler 2 kez convolution ve Relu işlemleri uygular
    conv_name_base = f'res{stage}{block}_branch'
    x = nn.Conv2d(input_tensor.shape[1], filters, kernel_size, padding=1)(input_tensor)
    x = nn.BatchNorm2d(filters)(x)
    x = F.relu(x, inplace=True)
    x = nn.Conv2d(filters, filters, kernel_size, padding=1)(x)
    x = nn.BatchNorm2d(filters)(x)
    #squeeze excitation bloğu eklenir
    x = se_block(x)
    #Resnet mimarisinde çıktılar birbirine sayısal olarak toplanır
    x = Add()([x, input_tensor])
    x = F.relu(x, inplace=True)
    return x

def conv_block(input_tensor, in_channels, out_channels, kernel_size, stride=2):
    #resnet mimarisinin temel bloğudur
    conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
    bn1 = nn.BatchNorm2d(out_channels)
    conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1)
    bn2 = nn.BatchNorm2d(out_channels)
    shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
    bn_shortcut = nn.BatchNorm2d(out_channels)

    def forward(x):
        residual = x
        out = conv1(x)
        out = bn1(out)
        out = F.relu(out, inplace=True)
        out = conv2(out)
        out = bn2(out)
        out = se_block(out)
        #oluşan her bir kanalın önem ağırlığını belirlemek için kullanılır
        #en önemli kanala daha fazla ağırlık vereerk nihai özellik vektörünü oluşturur
        shortcut_out = shortcut(out)
        shortcut_out = bn_shortcut(shortcut_out)
        out += shortcut_out
        out = F.relu(out, inplace=True)
        return out

    return forward


  
class ResNet18(nn.Module):
    #resnet 18 mimarisi identitiy ve conv katmanlarından oluşur
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = conv_block(64, 64, 3, stride=1)
        self.layer2 = identity_block(64, 128, 3)
        self.layer3 = identity_block(128, 256, 3)
        self.layer4 = identity_block(256, 512, 3)
        self.layer5 = identity_block(512, 512, 3)

    def forward(self, x):
        f1 = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(f1)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f3)
        f4 = self.layer4(f4)
        f5 = self.layer5(f5)
        return f1, f2, f3, f4, f5



