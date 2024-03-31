import torch 
import numpy as np
import argparse
import random
import math 
import os  
import torch.nn as nn 
import torch.optim as optim
import torchvision.transforms as transforms

from models import NetD, NetG, NetE 
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
parser.add_argument('--wtl2',  default=0.999, help='0 means don\'t use else use with this weight')
parser.add_argument('--useOverlapPred',  action='store_true', 
                    help='If true, it puts 10x more L2 weight on unmasked region.')
parser.add_argument('--nThreads',  default=4, help='# of data loading threads to use')
parser.add_argument('--niter',  type=int, default=25, help='# of iter at starting learning rate')
 
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
parser.add_argument('--checkpoint',  default='classifier.path.tar', help='checkpoint for saving/resuming models') 
parser.add_argument('--resume',  action='store_true', help='If use, resume model from checkpoint') 
parser.add_argument('--mask',  default='random', help='[random | square]') 
parser.add_argument('--loss_csv',  default='loss_data.csv', help='csv file for saving loss data') 
parser.add_argument('--acc_csv',  default='acc_data.csv', help='csv file for saving acc data') 
parser.add_argument('--pretrained_ckpt',  default='checkpoint.path.tar', help='checkpoint for loading pretrained weights') 
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr_step', default='40,60', help='decreasing strategy')
parser.add_argument('--update_csv', default='update_data.csv', help='data file for each update')
def main():
    opt = parser.parse_args()
    print(opt) 
         
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
               
    # prepare pretrained state_dict 
    nc = 3
    ngf = 64
    nz = 100 
    fineSize=128
    nef = 64  
    checkpoint = opt.pretrained_ckpt
    checkpoint = torch.load(checkpoint) 
    netG_state_dict = checkpoint['netG_state_dict']

    netE_state_dict = {}
    for name, param in netG_state_dict.items(): 
        if name.startswith('module.netE.'):
            netE_state_dict[name.replace('netE.', '0.')] = param 

    nBottleneck = 100 
    model = nn.Sequential(
        NetE(),
        nn.Flatten(),  # Flatten the output of NetE
        nn.Linear(nBottleneck, num_classes)
    )
    model = torch.nn.DataParallel(model).cuda() 
    for name, param in model.named_parameters():
        if name in netE_state_dict:
            # Load the corresponding weights from the checkpoint
            param.data = netE_state_dict[name]
            param.requires_grad = False
            pass
    
    # prepare loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda() 
    optimizer = torch.optim.SGD(model.parameters(), opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=120, gamma=0.1)

    # prepare csv files for trainning curves 
    with open(opt.acc_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'trainAcc', 'testAcc', 'trainLoss', 'testLoss'])
        
    with open(opt.update_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'k', 'Acc', 'Loss'])
 
    # training epoch 
    epoch_start = 0
    for ep in range(epoch_start, opt.niter): 
        # switch to train mode 
        model.train()
        for k, (data, target) in enumerate(train_loader, 0):   
            data = data.cuda()  
            target = torch.where(target == -1, torch.tensor(0), target) 
            target = target.cuda() 
            # compute output
            output = model(data)
            
            loss = criterion(output, target) 
            # compute acc 
            _, predicted = torch.max(output, dim=1) 
            
            n_correct = (predicted == target).sum().item()
            acc = n_correct / target.size(0)
 
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            print('[%d/%d][%d/%d] Acc: %.4f Loss: %.4f' 
              % (ep, opt.niter, k, len(train_loader), acc, loss.item())) 
            
            # store training data (acc, loss)
            with open(opt.update_csv, mode='a', newline='') as file:
                writer = csv.writer(file)  
                writer.writerow([ep, k, acc, loss.item()])
        
        # measure train and test accuracy and record loss  
        train_acc, train_loss = accuracy(train_loader, model, criterion)
        test_acc, test_loss = accuracy(val_loader, model, criterion)
        
        # store training data (acc, loss)
        with open(opt.acc_csv, mode='a', newline='') as file:
            writer = csv.writer(file)  
            writer.writerow([ep, train_acc, test_acc, train_loss, test_loss])

        # do checkpointing every 20 epochs (overwrites)  
        torch.save({'epoch':ep,
                    'state_dict':model.state_dict(), 
                    'optimizer':optimizer.state_dict(), 
                    'scheduler':scheduler.state_dict()}, 
                     opt.checkpoint)   

        
def accuracy(dataloader, model, criterion):
    model.eval()  # Set the model to evaluation mode
    total_samples = 0
    correct_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.cuda()
            labels = torch.where(labels == -1, torch.tensor(0), labels) 
            labels = labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, dim=1)
            total_samples += labels.size(0)
            correct_samples += (predicted == labels).sum().item()
            total_loss += loss.item() * labels.size(0)

    accuracy = correct_samples / total_samples
    loss = total_loss / total_samples

    return accuracy, loss
    
        
        
if __name__ == '__main__':
    main()