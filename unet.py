import torch
from torch import nn
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self, f=2, depth=5, padding=True, dropout=0., batchnorm=True):
        super().__init__()
        self.depth = depth
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        features = 1
        self.final = upblock(2**(f), 1, final=True, dropout=dropout,batchnorm=batchnorm,padding=padding)
        for i in range(depth):
            self.down.append(downblock(features, 2**(f+i), dropout=dropout, batchnorm=batchnorm, padding=padding))
            self.up.append(upblock(2**(f+i), features))
            features = 2**(f+i)
        self.lastact = nn.Sigmoid()
    def forward(self, x):
        skips = []
        for i in range(self.depth):
            skips.append(x.clone())
            x = self.down[i](x)
            x = F.max_pool2d(x, 2)
        for i in reversed(range(1,self.depth)):
            x = self.up[i](x, skips[i])
        x = self.final(x, skips[0])
        x = self.lastact(x)
        return x
        
class downblock(nn.Module):
    def __init__(self, insize, outsize, batchnorm=True, dropout=0.0, kernel_size=3, padding=True):
        super().__init__()
            
        block = [nn.Conv2d(insize, insize, kernel_size, padding=int(padding)), nn.LeakyReLU()]
        if dropout > 0.:
            block.append(nn.Dropout2d(dropout))
        if batchnorm:
            block.append(nn.BatchNorm2d(insize))
        block += [nn.Conv2d(insize, outsize, kernel_size, padding=int(padding)), nn.LeakyReLU()]
        if dropout > 0.:
            block.append(nn.Dropout2d(dropout))
        if batchnorm:
            block.append(nn.BatchNorm2d(outsize))
        self.block = nn.Sequential(*block)


    def forward(self, x):
        x = self.block(x)
        return x

class upblock(nn.Module):
    def __init__(self, insize, outsize, batchnorm=True, dropout=0.0, padding=True, kernel_size=3, final=False):
        super().__init__()
        if final:
            block = [nn.ConvTranspose2d(insize, 1, kernel_size, stride = 2)]
        else:
            block = [nn.ConvTranspose2d(insize, outsize, kernel_size, stride = 2)]
        if dropout > 0.:
            block.append(nn.Dropout2d(dropout))
        if batchnorm:
            block.append(nn.BatchNorm2d(outsize))
        if final:
            self.conv_block = downblock(2, 1, dropout=dropout, batchnorm=batchnorm, padding=padding)
        else:
            self.conv_block = downblock(insize, outsize, dropout=dropout, batchnorm=batchnorm, padding=padding)
        self.block = nn.Sequential(*block)
        
    def crop(self, input, desired_shape):
        current_shape = input.shape
        middle_x = int(input.shape[3] / 2)
        middle_t = int(input.shape[2] / 2)
        dif_x = int(desired_shape[3] / 2)
        dif_t = int(desired_shape[2] / 2)
        return input[:,:,(middle_t - dif_t): (middle_t + dif_t), (middle_x - dif_x): (middle_x + dif_x)]
        
    def forward(self, x, skip):
        x = self.block(x)
        if x.shape != skip.shape:
            x = self.crop(x, skip.shape)
        x = torch.cat([x, skip], 1)
        x = self.conv_block(x)
        return x
        
