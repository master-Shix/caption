"""
Created on Sun Dec 29 23:17:26 2019
https://github.com/alinstein/Depth_estimation/blob/master/Densenet_depth_model/model_dense.py
@author: alin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.depth.utils_depth.criterion import SiLogLoss
from models.depth._abstract import DepthModel


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        # interpolate x from x.size to larger size(concat_with.shape=(C,H,W))
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB(self.convB(self.leakyreluA(self.convA(torch.cat([up_x, concat_with], dim=1)))))


class Decoder(nn.Module):
    def __init__(self, num_features=2208, decoder_width=0.25):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSample(skip_input=features // 1 + 384, output_features=features // 2)
        self.up2 = UpSample(skip_input=features // 2 + 192, output_features=features // 4)
        #         self.up3 = UpSample(skip_input=features//4 +  96, output_features=features//8)
        self.up3 = UpSample(skip_input=features // 4 + 96, output_features=features // 16)
        self.up4 = UpSample(skip_input=features // 8 + 96, output_features=features // 16)

        self.conv3 = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[
            11]
        x_d0 = self.conv2(x_block4)
        # 15x20 to 15x20
        x_d1 = self.up1(x_d0, x_block3)
        # 15x20 to 30x40
        x_d2 = self.up2(x_d1, x_block2)
        # 30x40 to 60x80
        x_d3 = self.up3(x_d2, x_block1)
        # 60x80 to 120x160
        #         x_d4 = self.up4(x_d3, x_block0)
        # 120x160 to 240x320
        return self.conv3(x_d3)
    # return self.conv3(x_d4)


# Encoder uses the densenet_161 pretrained model
# following encoder encodes the image and store the features output from each output of layer
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        import torchvision.models as models
        self.original_model = models.densenet161(pretrained=True)

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items(): features.append(v(features[-1]))
        return features


class DenseModel(nn.Module):
    def __init__(self):
        super(DenseModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ADDED -- want to upsample to 512x512
class Model(nn.Module):
    def __init__(self, max_depth=10):
        super().__init__()
        self.dm = DenseModel()
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.max_depth = max_depth

    def forward(self, x):
        x = self.dm(x)
        x = self.up(x)
        x = self.up(x)
        x = torch.sigmoid(x) * self.max_depth
        return x


class SimpleBaseline(DepthModel):

    def __init__(self, max_depth=10, loss_type='SiLogLoss'):

        # order matters
        self.loss_type = loss_type
        super().__init__(max_depth)

    def initialize_model(self):
        print('initializing model')
        return Model(self.max_depth)

    def initialize_loss(self):
        if self.loss_type == 'SiLogLoss':
            # expects img1, img2 as parameters
            return SiLogLoss()
        else:
            raise ValueError('Invalid loss type {}'.format(self.loss_type))

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

