from resnet18 import resnet18
from squeezeexcitation import se_block
from keras.layers import *
from keras.models import *

class proposedNetwork(nn.Module):
    def __init__(self, height=256, width=256, num_class=1):
        super(proposedNetwork, self).__init__()

        #resnet 18 mimarisi
        self.resnet18 = resnet18()

        #Upsmapling işlemi U-net mimarisi yukarı besleme
        self.upconv1 = self._conv_block(512, 512)
        self.upconv2 = self._conv_block(512, 256)
        self.upconv3 = self._conv_block(256, 128)
        self.upconv4 = self._conv_block(128, 64)
        self.upconv5 = self._conv_block(64, 64)
        self.upconv6 = self._conv_block(64, 32)
        
        # Skip connections conv katmanları
        self.skip_conv4 = nn.Conv2d(64, 32, kernel_size=1)
        self.skip_conv5 = nn.Conv2d(64, 32, kernel_size=1)

        # Son katman
        self.final_conv = nn.Conv2d(96, num_class, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        #her bir u-net katmanında uygulanan işlemler
        #2 convolution işlemi ve normalizasyonlar
        #upsampling için kullanılır
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, img_input, mask_input=None):
        #resnet mimarisi çıkışları alınır
        #upsampling için beslenir
        f1, f2, f3, f4, f5 = self.resnet18(img_input)
        
        x = F.max_pool2d(f5, kernel_size=2, stride=2, padding=0)

        x = self.upconv1(x)
        x = se_block(x)
        x = torch.cat([x, f5], dim=1)
        
        x = self.upconv2(x)
        x = se_block(x)
        x = torch.cat([x, f4], dim=1)
        
        x = self.upconv3(x)
        x = se_block(x)
        x = torch.cat([x, f3], dim=1)
        
        x = self.upconv4(x)
        x = se_block(x)
        x = torch.cat([x, f2], dim=1)
        
        x = self.upconv5(x)
        
        skip4 = self.skip_conv4(x)
        skip4 = F.interpolate(skip4, scale_factor=4, mode='bilinear', align_corners=True)
        
        skip5 = self.skip_conv5(x)
        #interpolate fonksiyonu tensör byutlarını yeniden ölçeklendirir
        #çarpma katsayısı, ölçeklendirme faktörü =  2
        #4d ölçeklendirme yap
        #align_corners = true -> giriş ve çıkış kenarlarını hizala
        skip5 = F.interpolate(skip5, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.upconv6(x)
        #Cat işlemi piksel bazında toplama yapar
        x = torch.cat([skip4, skip5, x], dim=1)
        
        output = self.final_conv(x)
        output = torch.sigmoid(output)

        return output

# Model çıktısı segmente görüntüyü, yani maskeyi verir
model = ProposedNetwork()