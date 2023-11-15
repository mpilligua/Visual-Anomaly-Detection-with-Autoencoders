import torch
import torch.nn as nn

class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, 3, stride=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.conv2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=1, return_indices=True)
        self.indices1 = 0
        self.indices2 = 0

        # Decoder
        self.unpool1 = nn.MaxUnpool2d(2, stride = 1)
        self.unconv1 = nn.ConvTranspose2d(8, 16, 3, stride = 2, padding = 1) 
        self.unpool2 = nn.MaxUnpool2d(2, stride = 2)
        self.unconv2 = nn.ConvTranspose2d(16, 3, 3, stride = 3, padding = 1)

    def encoder(self, x):
        x = torch.relu(self.conv1(x))
        x, self.indices1 = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x, self.indices2 = self.pool2(x)
        return x

    def decoder(self, x):
        x = self.unpool1(x, self.indices2)
        x = torch.relu(self.unconv1(x))
        x = self.unpool2(x, self.indices1)
        x = torch.sigmoid(self.unconv2(x))
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 255 x 255 x 3
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)# output: 255x255x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 255x255x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 128x128x64

        # input: 128 x 128 x 64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 128x128x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 128x128x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 64x64x128

        # input: 64x64x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 64x64x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 64x64x256

        # Decoder
        #input: 64x64x256
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # output: 128x128x128
        
        # input 256x256x128 because of skip connection

        # The plus is because of the skip conection
        self.d11 = nn.Conv2d(128+128, 128, kernel_size=3, padding=1) # output: 128x128x128
        self.d12 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 128x128x128
        
        # input 128x128x128
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # output: 256x256x64

        # The plus is because of the skip conection
        self.d21 = nn.Conv2d(64+64, 64, kernel_size=3, padding=1) # output: 255x255x64
        self.d22 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 255x255x64

        # Output layer
        # input 255x255x64
        self.outconv = nn.Conv2d(64, 3, kernel_size=1)
    
    def forward(self, x, return_embedding=False):
        # Encoder
        xe11 = torch.relu(self.e11(x))
        xe12 = torch.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = torch.relu(self.e21(xp1))
        xe22 = torch.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = torch.relu(self.e31(xp2))
        xe32 = torch.relu(self.e32(xe31))
  
        # Decoder
        xu1 = self.upconv1(xe32)
        xu11 = torch.cat([xu1, xe22], dim=1)
        xd11 = torch.relu(self.d11(xu11))
        xd12 = torch.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe12], dim=1)
        xd21 = torch.relu(self.d21(xu22))
        xd22 = torch.relu(self.d22(xd21))

        # Output layer
        out = self.outconv(xd22)

        if return_embedding:
            return out, xe32
        else:
            return out

class UNet_NotRes(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 255 x 255 x 3
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)# output: 255x255x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 255x255x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 128x128x64

        # input: 128 x 128 x 64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 128x128x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 128x128x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 64x64x128

        # input: 64x64x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 64x64x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 64x64x256

        # Decoder
        #input: 64x64x256
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # output: 128x128x128
        
        self.d11 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 128x128x128
        self.d12 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 128x128x128
        
        # input 128x128x128
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # output: 256x256x64

        self.d21 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 255x255x64
        self.d22 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 255x255x64

        # Output layer
        # input 255x255x64
        self.outconv = nn.Conv2d(64, 3, kernel_size=1)
    
    def forward(self, x, return_embedding=False):
        # Encoder
        xe11 = torch.relu(self.e11(x))
        xe12 = torch.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = torch.relu(self.e21(xp1))
        xe22 = torch.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = torch.relu(self.e31(xp2))
        xe32 = torch.relu(self.e32(xe31))

        # Decoder
        xu1 = self.upconv1(xe32)
        xd11 = torch.relu(self.d11(xu1))
        xd12 = torch.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xd21 = torch.relu(self.d21(xu2))
        xd22 = torch.relu(self.d22(xd21))

        # Output layer
        out = self.outconv(xd22)

        if return_embedding:
            return out, xd32
        else:
            return out


class Unet_preconstructed(nn.Module):
    def __init__(self):
        super().__init__()
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=False)
    
    def forward(self, x):
        return model(x)


class MLP_emb(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_emb, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x


class extended_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x256

        # input: 32x32x256
        self.e41 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 16x16x128
        self.e51 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 8x8x64
        self.e61 = nn.Conv2d(64, 16, kernel_size=3, padding=1)
        self.e62 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 4x4x16
        self.e71 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.e72 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        # 8 x 8 x 3

        # output: 4x4x3
        self.emb = nn.Linear(4*4*3, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        xp3 = self.pool3(x)

        xe41 = torch.relu(self.e41(xp3))
        xe42 = torch.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = torch.relu(self.e51(xp4))
        xe52 = torch.relu(self.e52(xe51))
        xp5 = self.pool5(xe52)

        xe61 = torch.relu(self.e61(xp5))
        xe62 = torch.relu(self.e62(xe61))
        xp6 = self.pool6(xe62)

        xe71 = torch.relu(self.e71(xp6))
        xe72 = torch.relu(self.e72(xe71))

        # print(xe72.shape)
        flatten = xe72.view(-1, 4*4*3)
        # print(flatten.shape)
        emb = self.emb(flatten)
        # print(emb.shape)
        emb = self.softmax(emb)
        # print(emb.shape)

        return emb

class unet_cnn(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.unet = pretrained
        self.cnn = extended_CNN()

    def forward(self, x):
        x, emb = self.unet(x, return_embedding = True)
        x = self.cnn(emb)
        return x
