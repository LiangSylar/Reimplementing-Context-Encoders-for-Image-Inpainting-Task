import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
import random
import pandas as pd
from torchvision.io import read_image
from PIL import Image

_NUM_CLASSES = {
    'cifar10': 10,
    'celeba': 2,
    'gtsrb': 43,
}


def cifar10(batch_size, data_root='./datasets', train=True, val=True, fineSize=32, portion=0.5, **kwargs):
#     data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    data_root = os.path.join(data_root, 'cifar10') 
    num_workers = kwargs.setdefault('num_workers', 4)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Scale(fineSize),
#                     transforms.Pad(4),
#                     transforms.RandomCrop(fineSize),
#                     transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        print("CIFAR-10 training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.Scale(fineSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
        print("CIFAR-10 testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def celeba(batch_size, data_root='./datasets', train=True, val=True,
           fineSize=32, portion=0.5, test_split=0.2, **kwargs):
    data_root = os.path.join(data_root, 'img_align_celeba')
    label_csv = os.path.join(data_root, 'list_attr_celeba.csv')
    img_root = os.path.join(data_root, 'img_align_celeba')
    
    num_workers = kwargs.setdefault('num_workers', 4) 
    print("Building celebA data loader with {} workers".format(num_workers))
    ds = [] 
    # define data transformations
    transform = transforms.Compose([
        transforms.Resize((fineSize, fineSize)),  # Resize the images to a desired size
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image tensors
    ])
    # Create datasets and data loaders for train and test sets
    train_dataset = celeba_dataset(label_csv, img_root, transform)
    test_dataset = celeba_dataset(label_csv, img_root, transform, train=False)
    
    if train:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, 
            shuffle=True, drop_last=True, num_workers=num_workers
        )
        print("CelebA training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, 
            shuffle=False, drop_last=True, num_workers=num_workers
        )
        print("CelebA testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
        
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def gtsrb(batch_size, data_root='./datasets', train=True, val=True,
           fineSize=32, portion=0.5, test_split=0.2, **kwargs): 
    data_root = os.path.join(data_root, 'gtsrb')  
    label_csv = os.path.join(data_root, 'Test.csv')
    num_workers = kwargs.setdefault('num_workers', 4) 
    print("Building celebA data loader with {} workers".format(num_workers))
    ds = [] 
    transform = transforms.Compose([ 
        transforms.Resize((fineSize, fineSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
    ])
    
    if train: 
        train_data_root = os.path.join(data_root, 'train')
        train_loader =  torch.utils.data.DataLoader(
            datasets.ImageFolder(train_data_root,
                                 transform=transform),
            batch_size=batch_size, shuffle=True, drop_last=True, 
            num_workers=num_workers,
        )
        print("GTSRB training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader) 
        
    if val:
        val_data_root = os.path.join(data_root, '')
        val_label_root = os.path.join(data_root, 'Test.csv')
        val_loader =  torch.utils.data.DataLoader(
            GTSRB_test(val_label_root, val_data_root, transform=transform),
            batch_size=batch_size, shuffle=False, drop_last=True,
            num_workers=num_workers, 
        )
        print("GTSRB testing data size: {}".format(len(val_loader.dataset)))
        ds.append(val_loader)

    ds = ds[0] if len(ds) == 1 else ds
    return ds

class celeba_dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, attribute='Brown_Hair', train=True):  
        self.img_dir = img_dir
        self.transform = transform 
        # extrac specific attribute from csv label file
        df = pd.read_csv(annotations_file)    
        self.img_labels = df[['image_id', attribute]]
        
        if train: # train dataset: selects first 80,000
            self.img_labels = df.loc[:90000]
        else: # test dataset: selects 80,001 ~ 100,000
            self.img_labels = df.loc[90000:100000]
    
    def __len__(self): 
        return len(self.img_labels)-1 
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1] 
        if self.transform:
            image = self.transform(image)
        return image, label 
    
class GTSRB_test(Dataset): 

    def __init__(self, annotations_file, img_dir, transform=None):   
        self.img_dir = img_dir  
        self.transform = transform
        df = pd.read_csv(annotations_file)
        self.img_labels = df[['Path', 'ClassId']] 

    def __len__(self): 
        return len(self.img_labels) 
    
    def __getitem__(self, idx):  
         
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]   
        if self.transform:
            image = self.transform(image)
        return image, label 

