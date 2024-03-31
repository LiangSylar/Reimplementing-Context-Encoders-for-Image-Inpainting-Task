import torch
import torch.nn as nn
import torch.nn.functional as F
 

__all__ = ['NetG', 'netD']
    
class NetE(nn.Module):
    def __init__(self, nc=3, nef=64, fineSize=128, nBottleneck=100, opt=None):
        super(NetE, self).__init__()
        if opt != None:
            nc = opt.nc # number of channels of input
            nef = opt.nef # number of encoder filters 
            self.fineSize = opt.fineSize 
            self.nBottleneck = opt.nBottleneck
        else:
            nc = nc
            nef = nef
            self.fineSize = fineSize
            self.nBottleneck = nBottleneck
            
        self.conv1 = nn.Conv2d(nc, nef, kernel_size=4, stride=2, padding=1, bias=False)
        self.leakyrelu1 = nn.LeakyReLU(0.2, inplace=True)

        if self.fineSize == 128:
            # state size: (nef) x 64 x 64
            self.conv2 = nn.Conv2d(nef, nef, 4, 2, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(nef)
            self.leakyrelu2 = nn.LeakyReLU(0.2, inplace=True)
            
        # state size: (nef) x 32 x 32
        self.conv3 = nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(nef * 2)
        self.leakyrelu3 = nn.LeakyReLU(0.2, inplace=True)
        
        # state size: (nef) x 16 x 16
        self.conv4 = nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(nef * 4)
        self.leakyrelu4 = nn.LeakyReLU(0.2, inplace=True)
        
        # state size: (nef) x 8 x 8
        self.conv5 = nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(nef * 8)
        self.leakyrelu5 = nn.LeakyReLU(0.2, inplace=True)
        
        # state size: (nef*8) x 4 x 4
        self.conv6 = nn.Conv2d(nef * 8, nBottleneck, 4, 1, 0, bias=False)
        # state size: (nBottleneck) x 1 x 1

    def forward(self, input):
        # input is (nc) x 128 x 128
        x = self.conv1(input)
        x = self.leakyrelu1(x)
        # state size: (nef) x 64 x 64
        if self.fineSize == 128:
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.leakyrelu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyrelu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leakyrelu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.leakyrelu5(x)

        x = self.conv6(x)
        # state size: (nBottleneck) x 1 x 1
        return x
    

class NetG(nn.Module):
    def __init__(self, nc=3, ngf=64, nz=100, fineSize=128, nef=64, nBottleneck=100, opt=None, noiseGen=False):
        super(NetG, self).__init__()
        if opt!= None:
            nc = opt.nc # number of input channels 
            ngf = opt.ngf # number of generator filters 
            nz = opt.nz # number of dim for noise Z
            self.fineSize = opt.fineSize # int
            self.noiseGen = opt.noiseGen # boolean
            self.netE = NetE(opt=opt) # module instance (encoder network)
            self.nBottleneck = opt.nBottleneck
        else:
            nc = nc # number of input channels
            ngf = ngf # number of generator filters 
            nz = nz # number of dim for noise Z
            self.fineSize = fineSize # int
            self.noiseGen = noiseGen # boolean
            self.netE = NetE(nc=nc, nef=nef, fineSize=fineSize) # module instance (encoder network)
            self.nBottleneck = nBottleneck
            
        if self.noiseGen:
            self.netG_noise = nn.Sequential(
                nn.Conv2d(nz, nz, 1, 1, 0, bias=False)
            )   
            self.netG = nn.Sequential( 
                nn.BatchNorm2d(self.nBottleneck + nz),
                nn.LeakyReLU(0.2, inplace=True)
            )
            nz_size = self.nBottleneck + nz
        else:
            self.netG = nn.Sequential( 
                nn.BatchNorm2d(self.nBottleneck),
                nn.LeakyReLU(0.2, inplace=True)
            ) 
            nz_size = self.nBottleneck

        self.conv1 = nn.ConvTranspose2d(nz_size, ngf * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4)
        self.relu2 = nn.ReLU(True)

        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        self.relu4 = nn.ReLU(True)

        if self.fineSize  == 128:
            self.conv5 = nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False)
            self.bn5 = nn.BatchNorm2d(ngf)
            self.relu5 = nn.ReLU(True)

        self.conv6 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, input, Z=None):
        x = self.netE(input)
        if self.noiseGen:
            Z_conv = self.netG_noise(Z)
            x = torch.cat((x, Z_conv), 1) # concatenate by channel 
        x = self.netG(x) 
        
        # input is Z: (nz_size) x 1 x 1, going into a convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # state size: (ngf*8) x 4 x 4
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # state size: (ngf*4) x 8 x 8
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        # state size: (ngf*2) x 16 x 16 
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        # state size: (ngf) x 32 x 32
        if self.fineSize == 128:
            x = self.conv5(x)
            x = self.bn5(x)
            x = self.relu5(x)
            # state size: (ngf) x 64 x 64

        x = self.conv6(x)
        x = self.tanh(x)
        # state size: (nc) x 128 x 128

        return x
    
class NetD(nn.Module):
    def __init__(self, opt):
        super(NetD, self).__init__()
        nc = opt.nc
        ndf = opt.ndf
        self.conditionAdv = opt.conditionAdv
        self.fineSize = opt.fineSize

        if self.conditionAdv:
            print('conditional adv not implemented')
            exit()
            # Code for conditional adversarial not implemented

        # input is (nc) x 128 x 128, going into a convolution
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.leakyrelu1 = nn.LeakyReLU(0.2, inplace=True)

        if opt.fineSize == 128:
            self.conv2 = nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(ndf)
            self.leakyrelu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 2)
        self.leakyrelu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf * 4)
        self.leakyrelu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(ndf * 8)
        self.leakyrelu5 = nn.LeakyReLU(0.2, inplace=True)

        self.conv6 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Flatten the output
        self.flatten = nn.Flatten(1, -1)

    def forward(self, input, input_ctx=None):
        '''
        input: input_center, real images if train with real; 
               fake images (output of netG) if train with fake.
        input_ctx: original data with regions masked and filled with channel mean.
        '''
        x = self.conv1(input)
        x = self.leakyrelu1(x)

        if self.fineSize == 128:
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.leakyrelu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyrelu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leakyrelu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.leakyrelu5(x)

        x = self.conv6(x)
        x = self.sigmoid(x)
#         print("sigmoid:", x.shape) # [64, 1, 1, 1]
        # state size: 1 x 1 x 1 
        
        x = self.flatten(x) 
#         print("flatten", x.shape) # [64, 1]
        x = x.squeeze(1)
#         print("squeeze", x.shape) # [64]
        # state size: 1
        return x 
      