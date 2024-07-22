import torch
import torch.nn as nn

# Rappresenta un layer di convoluzione standard con dropout 
class Conv(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, 3, 1, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout_prob)  
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)  
        return x

# Rappresenta un layer convoluzionale con pooling e dropout per il percorso di downsampling
class Down(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, 3, 1, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout_prob)  
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)  
        x = self.pool(x)
        return x

# Rappresenta un layer convoluzionale con pooling e dropout per il percorso di upsampling
class Up(nn.Module):
    def __init__(self, in_features, skip_features, out_features, dropout_prob=0.0):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_features, in_features, 4, 2, 1)
        self.conv = nn.Conv2d(in_features + skip_features, out_features, 3, 1, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout_prob)  # Dropout2d per dropout nei canali
        
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)  
        return x


class Model(nn.Module):
    def __init__(self, dropout_prob=0.0):
        super().__init__()
        # Downsampling path
        self.down_0 = Conv(1, 64, dropout_prob)
        self.down_1 = Down(64, 128, dropout_prob)
        self.down_2 = Down(128, 128, dropout_prob)
        self.down_3 = Down(128, 256, dropout_prob)
        self.down_4 = Down(256,512,dropout_prob)
        # Bottleneck
        self.bottleneck = Conv(512, 512, dropout_prob)
        # Upsampling path
        self.up_1 = Up(512, 256, 256,dropout_prob) # skip from down_3
        self.up_2 = Up(256, 128, 128, dropout_prob) # skip from down_2
        self.up_3 = Up(128, 128, 128, dropout_prob) # skip from down_1
        self.up_4 = Up(128, 64, 64, dropout_prob)  # skip from down_0
        # Classificatore (senza funzione di attivazione perche viene usata in uscita , kernel size: 1)
        self.classifier = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        x_0 = self.down_0(x)   # 256x256
        x_1 = self.down_1(x_0) # 128x128
        x_2 = self.down_2(x_1) # 64x64
        x_3 = self.down_3(x_2) # 32x32
        x_4 = self.down_4(x_3) # 16x16
        # Bottleneck
        x_5 = self.bottleneck(x_4) # 16x16
        # Upsampling path
        x_6 = self.up_1(x_5, x_3) # 32x32
        x_7 = self.up_2(x_6, x_2) # 64x64
        x_8 = self.up_3(x_7, x_1) # 128x128
        x_9 = self.up_4(x_8, x_0) # 256x256
        # Classificatore
        x_10 = self.classifier(x_9)
        return x_10
