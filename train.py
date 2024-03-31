import torch 
import numpy as np
import argparse
import random
import math 
import os  
import torch.nn as nn 
import torch.optim as optim
import torchvision.transforms as transforms

from models import NetD, NetG 
import datasets

import csv

# parse arguments 
parser = argparse.ArgumentParser(description='PyTorch Context-Encoders Training')
parser.add_argument('--batch_size',  type=int, default=128, help='number of samples to produce')
parser.add_argument('--loadSize',  default=350, 
                    help="resize the loaded image to loadsize maintaining aspect ratio. \
                    0 means Don't resize.")
parser.add_argument('--fineSize',  default=128, type=int, 
                    help='size of random crops. Only 64 and 128 supported.')
parser.add_argument('--nBottleneck',  default=100, help='#of dim for bottleneck of encoder')
parser.add_argument('--nef',  default=64, help='#of encoder filters in first conv layer')
parser.add_argument('--ngf',  default=64, help='#of geb filters iin first conv layer')
parser.add_argument('--ndf',  default=64, help='#of discrim filters in first conv layer')
parser.add_argument('--nc',  default=3, help='#of channels in input')
parser.add_argument('--wtl2',  type=int, default=0.999, help='0 means don\'t use else use with this weight')
parser.add_argument('--useOverlapPred',  action='store_true', 
                    help='If true, it puts 10x more L2 weight on unmasked region.')
parser.add_argument('--nThreads',  default=4, help='# of data loading threads to use')
parser.add_argument('--niter',  type=int, default=25, help='# of iter at starting learning rate')
parser.add_argument('--lr',  default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1',  default=0.5, help='momentum term of adam')
parser.add_argument('--ntrain',  default=math.inf, 
                    help='# of examples per epoch. math.inf for full dataset')
parser.add_argument('--display',  action='store_true', 
                    help='display samples while training.')
parser.add_argument('--display_id', default=10, 
                    help='display window id')
parser.add_argument('--display_iter', default=50, 
                    help='# iterations after which display is updated')
# parser.add_argument('--gpu', action='store_true', 
#                     help='gpu=0 is CPU mode. gpu=X is GPU mode on GPU X.')
parser.add_argument('--gpu', default='0', help='index of gpus to use')
parser.add_argument('--name', default='train1', 
                    help='name of the experiment you are running.')
parser.add_argument('--manualSeed', default=0, help='0 means random seed')
# extra options 
parser.add_argument('--conditionAdv', action='store_true', help='use conditional Adv')
parser.add_argument('--noiseGen', action='store_true', help='use noiseGen')
parser.add_argument('--noiseType', default='normal', help='uniform / normal')
parser.add_argument('--nz', default=100, help='# of dim for noise vector Z') 
# new options
parser.add_argument('--dataset',  default='cifar10', help='cifar10 / lsun')
parser.add_argument('--data_root',  default='../datasets', help='cifar10') 
parser.add_argument('--checkpoint',  default='checkpoint.path.tar', help='checkpoint for saving/resuming models') 
parser.add_argument('--resume',  action='store_true', help='If use, resume model from checkpoint') 
parser.add_argument('--mask',  default='random', help='[random | square]') 
parser.add_argument('--loss_csv',  default='loss_data.csv', help='csv file for saving loss data') 

def main():
    opt = parser.parse_args()
    print(opt)
    
    # prepare directories 
      # checkpoint directory (model)
      # masked images 
      # reconstructed images 
         
    # set seed
    if opt.manualSeed == 0:
        opt.manualSeed = random.randint(1, 10000) 
    print("Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.gpu:
        torch.cuda.manual_seed_all(opt.manualSeed) 
    torch.set_num_threads(1) # Sets the number of threads used for intraop parallelism on CPU. 
    
    # select gpus
    opt.gpu = opt.gpu.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(opt.gpu)
     
    # data loader
    datasize = None
    num_classes = None
    train_loader = None
    val_loader = None 
    if opt.dataset in ['cifar10', 'celeba', 'gtsrb']:
        assert callable(datasets.__dict__[opt.dataset])
        get_dataset = getattr(datasets, opt.dataset)
        num_classes = datasets._NUM_CLASSES[opt.dataset]
        train_loader, val_loader = get_dataset(
            batch_size=opt.batch_size, num_workers=opt.nThreads, 
            data_root=opt.data_root, fineSize = opt.fineSize) 
    else:
        raise Exception("Unsupported datasets!") 
             
    # initialize variables
    nc = opt.nc
    nz = opt.nz
    nBottleneck = opt.nBottleneck
    ndf = opt.ndf
    ngf = opt.ngf
    nef = opt.nef
    real_label = 1
    fake_label = 0
    
    # set up models: netG and netD
    netG = NetG(opt=opt)
    netD = NetD(opt=opt)  
    
    # set up optimizer: adam 
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
     
        
    # initialize netG and netD with custom initialization method 
    netG.apply(weights_init) 
    netD.apply(weights_init) 
      
    # create loss functions 
    criterion = nn.BCELoss().cuda()
    criterionMSE = nn.MSELoss().cuda()  
    
    # Generating random pattern
    res = 0.06 # the lower it is, the more continuous the output will be. 0.01 is too small and 0.1 is too large
    density = 0.25
    MAX_SIZE = 10000  
    low_pattern = torch.FloatTensor(int(res * MAX_SIZE), int(res * MAX_SIZE)).uniform_(0, 1).mul(255)  
#     print(low_pattern.shape) # [600, 600] 
    low_pattern = low_pattern.unsqueeze(0)   # Reshape to have 2 dimensions
#     print(low_pattern.shape)  # [1, 600, 600]
    # Resize the low-resolution pattern to the desired size
    transform = transforms.Resize((MAX_SIZE, MAX_SIZE), interpolation=transforms.InterpolationMode.BICUBIC)
    pattern = transform(low_pattern)
#     print(pattern.shape)  # [1, 10000, 10000] 
    pattern = pattern.squeeze(0)
#     print(pattern.shape) # [10000, 10000] 
    pattern.div_(255)
    pattern = torch.lt(pattern, density).byte()  # 25% 1s and 75% 0s
    pattern = pattern.byte()
    pattern = pattern.bool()
    print('...Random pattern generated')
    
    ## prepare variables in advance for the convenience of loading data to cuda  
    label = torch.FloatTensor(opt.batch_size) 
    # input_ctx: input to netG
    input_ctx = torch.FloatTensor(opt.batch_size, 3, opt.fineSize, opt.fineSize)
    # input_center: input to netD
    input_center = torch.FloatTensor(opt.batch_size, 3, opt.fineSize, opt.fineSize) 
    # noise: another input to netG is opt.noiseGen is on.
    noise = torch.FloatTensor(opt.batch_size, opt.nz, 1, 1)
    # input_real_center: input to criterionMSE
    input_real_center = torch.FloatTensor(opt.batch_size, 3, opt.fineSize, opt.fineSize) 
    
    # load model and data to cuda  
    netG = torch.nn.DataParallel(netG, device_ids=range(len(opt.gpu))).cuda() 
    netD = torch.nn.DataParallel(netD, device_ids=range(len(opt.gpu))).cuda() 
    criterion.cuda()
#     criterionMSE.cuda()
    label = label.cuda()
    input_ctx, noise = input_ctx.cuda(), noise.cuda()
    input_center = input_center.cuda() 
    if opt.wtl2 != 0:
        criterionMSE.cuda() 
        input_real_center = input_real_center.cuda()
                    
    # resume training if required 
    epoch_start = 0
    if opt.resume:
        checkpoint = torch.load(opt.checkpoint)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        epoch_start = checkpoint['epoch']+1
        
    # prepare csv files for trainning curves
    if not opt.resume: 
        with open(opt.loss_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'errD', 'errG', 'errG_adv', 'errG_l2'])
    
    print(opt.wtl2)
    # training epoch 
    for ep in range(epoch_start, opt.niter):
        for k, data in enumerate(train_loader, 0): 
            # prepare data copies 
            real_ctx, _ = data
            real_center = real_ctx  # view
            input_center = real_center.clone().cuda() 
#             input_center.data.resize_(real_center.size()).copy_(real_center)
    
            if opt.wtl2 !=0: 
                input_real_center = real_center.clone().cuda()   
#                 input_real_center.data.resize_(real_center.size()).copy_(real_center)
                
            # Get random mask
            wastedIter = 0
            while True:
                x = random.uniform(1, MAX_SIZE - opt.fineSize)
                y = random.uniform(1, MAX_SIZE - opt.fineSize)
                mask = pattern[int(y):int(y + opt.fineSize), int(x):int(x + opt.fineSize)]  # view, no allocation
                area = mask.sum().item() * 100. / (opt.fineSize * opt.fineSize)
                if 20 < area < 30:  # want it to be approximately 75% 0s and 25% 1s
                    # print('wasted tries: ', wastedIter)
                    break
                wastedIter += 1      
            mask_global = mask.repeat(opt.batch_size, 1, 1) # [batch_size, 128, 128] 
#             print(mask)
            
            # Apply mask to real_ctx 
#             print(mask_global.shape)
#             print(real_ctx.shape) # [64, 3, 32, 32]

            real_ctx[:, 0, :, :].masked_fill_(mask_global, 2 * 117.0 / 255.0 - 1.0) 
            real_ctx[:, 1, :, :].masked_fill_(mask_global, 2 * 104.0 / 255.0 - 1.0) 
            real_ctx[:, 2, :, :].masked_fill_(mask_global, 2 * 123.0 / 255.0 - 1.0)  
            
            # make data copies 
#             input_ctx = real_ctx.clone()
            input_ctx.data.resize_(real_ctx.size()).copy_(real_ctx)
             
            # set train mode
            netG.train()
            netD.train() 
            
            # train netD with real 
            netD.zero_grad()
            label.data.resize_(opt.batch_size).fill_(real_label)
#             if opt.conditionAdv:
#                 output = netD(input_ctx, input_center)
#             else:
#                 output = netD(input_center)
            output = netD(input_center)
#             print(output.shape, label.shape) # [64, 1]; [64]
            errD_real = criterion(output, label) 
            errD_real.backward()            
      
            # train netD with fake 
            if opt.noiseType == 'uniform':  
                noise_ = torch.Tensor(opt.batch_size, opt.nz, 1, 1).uniform_(-1, 1) 
                noise.data.resize_(noise_.size()).copy_(noise_)
            elif opt.noiseType == 'normal': 
                noise_ = torch.Tensor(opt.batch_size, opt.nz, 1, 1).normal_(0, 1) 
                noise.data.resize_(noise_.size()).copy_(noise_)
            if opt.noiseGen:
                fake = netG(input_ctx, noise)
            else:
                fake = netG(input_ctx)  
            output = netD(fake.detach()) # why detach here - solved.  
            label.data.resize_(opt.batch_size).fill_(fake_label)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()
     
            # train netG 
            netG.zero_grad()
            label.data.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake) 
            errG_adv = criterion(output, label)   
             
            errG = 0 
            if opt.wtl2 != 0:
                errG_l2 = (fake-input_real_center).pow(2)
#                 errG_l2 = criterionMSE(input_center, input_real_center)  
                overlapL2Weight = 10 
                wtl2Matrix = fake.clone().fill_(overlapL2Weight * opt.wtl2)  
                
#                 print(wtl2Matrix.is_cuda, mask_global.is_cuda) # T, F
                mask_global = mask_global.cuda()
                for i in range(3):
                    wtl2Matrix[:, i, :, :].masked_fill_(mask_global, opt.wtl2)
                    
                errG_l2 = errG_l2 * wtl2Matrix  
                errG_l2 = errG_l2.sum() 
                if 0 < opt.wtl2 <= 1:             
                    errG = (1 - opt.wtl2) * errG_adv + opt.wtl2 * errG_l2
                else: 
                    errG = errG_adv + opt.wtl2 * errG_l2
                
            errG.backward()
            optimizerG.step() 
            print('[%d/%d][%d/%d] errD: %.4f errG: %.4f errG_adv: %.4f errG_l2: %.4f'
              % (ep, opt.niter, k, len(train_loader),
                 errD.item(), errG.item(), errG_adv.item(), errG_l2.item(), )) 
            
        with open(opt.loss_csv, mode='a', newline='') as file:
            writer = csv.writer(file)  
            writer.writerow([ep, errD.item(), errG.item(), errG_adv.item(), errG_l2.item()])

        # do checkpointing every epoch (overwrites) 

        torch.save({'epoch':ep,
                    'netG_state_dict':netG.state_dict(),
                    'netD_state_dict':netD.state_dict(),
                    'optimizerG_state_dict':optimizerG.state_dict(), 
                    'optimizerD_state_dict':optimizerD.state_dict()}, 
                     opt.checkpoint)  
                
    
def weights_init(m):
    '''
    Implement weights initialization methods.
    This method applies on both netG and netD. 
    '''
    classname = m.__class__.__name__
    if 'Conv' in classname:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif 'BatchNorm' in classname:
        if m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    
     


if __name__ == '__main__':
    main()