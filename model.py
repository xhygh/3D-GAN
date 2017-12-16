import torch
import torch.optimizer as optim
from torch.autograd import variable

import torch.nn.founctional as F
import numpy as np
import torch.nn import nn
class Generator(nn.Module):
    def __init__(self):
        nz = 200
        ngf = 64
        super(Generator, self).__init__(
            dc1= nn.ConvTranspose3d(in_channels=nz, out_channels=ngf * 8, kernel_size=4, stride=1, padding=0),
            dc2= nn.ConvTranspose3d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1),
            dc3= nn.ConvTranspose3d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1),
            dc4= nn.ConvTranspose3d(in_channels=ngf * 2, out_channels=ngf * 1, kernel_size=4, stride=2, padding=1),
            dc5= nn.ConvTranspose3d(in_channels=ngf , out_channels= 1, kernel_size=4, stride=2, padding=1),
            #############
            bn1 = nn.BatchNorm3d(ngf * 8),
            bn2 = nn.BatchNorm3d(ngf * 4),
            bn3 = nn.BatchNorm3d(ngf * 2),
            bn4 = nn.BatchNorm3d(ngf),
            bn5 = nn.BatchNorm3d(1),
        )

    def __call__(self, z):
        y = F.relu(self.bn1(self.dc1(z)))
        y = F.relu(self.bn2(self.dc2(y)))
        y = F.relu(self.bn3(self.dc3(y)))
        y = F.relu(self.bn4(self.dc4(y)))
        y = F.sigmoid(self.bn5(self.dc5(y)))

        return y
    
    def make_hidden(self, batchsize, train=True):
        if train:
            return np.random.normal(0, 0.33, size=[batchsize, 200, 1, 1, 1]).astype(np.float32)
        else:
            return np.random.uniform(-1, 1, (batchsize, 200, 1, 1, 1)).astype(np.float32)

class Discriminator(chainer.Chain):
    def __init__(self):
        ngf = 64
        super(Discriminator, self).__init__(
            dc1= nn.ConvTranspose3d(in_channels=1, out_channels=ngf , kernel_size=4, stride=2, padding=1),
            dc2= nn.ConvTranspose3d(in_channels=ngf, out_channels=ngf * 2 , kernel_size=4, stride=2, padding=1),
            dc3= nn.ConvTranspose3d(in_channels=ngf*2, out_channels=ngf * 4 , kernel_size=4, stride=2, padding=1),
            dc4= nn.ConvTranspose3d(in_channels=ngf*4, out_channels=ngf * 8 , kernel_size=4, stride=2, padding=1),
            dc5= nn.ConvTranspose3d(in_channels=ngf*8, out_channels=1 , kernel_size=4, stride=1, padding=1),
            #############
            bn1 = nn.BatchNorm3d(ngf),
            bn2 = nn.BatchNorm3d(ngf * 2),
            bn3 = nn.BatchNorm3d(ngf * 4),
            bn4 = n.BatchNorm3d(ngf * 8),
            bn5 = nn.BatchNorm3d(1)
        )

    def __call__(self, x):
        y = F.leaky_relu(self.bn1(self.dc1(x)), slope=0.2)
        y = F.leaky_relu(self.bn2(self.dc2(y)), slope=0.2)
        y = F.leaky_relu(self.bn3(self.dc3(y)), slope=0.2)
        y = F.leaky_relu(self.bn4(self.dc4(y)), slope=0.2)
        y = F.sigmoid(self.bn5(self.dc5(y)))

        return y

def count_model_params(m):
    return sum(p.data.size for p in m.params())

def main():
    m = Generator()
    print(count_model_params(m))
    m = Discriminator()
    print(count_model_params(m))

if __name__=="__main__":
    main()
