
import torch
import math
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from setting import get_parser
import numpy as np
#======================= utils ==========================================
parser = get_parser()
args = parser.parse_args()

def evalaa(model, testdata):
    with torch.no_grad():
        total = 0
        top1 = 0
        topk = 0
        for (test_imgs, test_labels) in tqdm(testdata):
            test_labels = test_labels.to(args.device)
            out, _, _, _, _ = model(test_imgs.to(args.device))
            _,maxk = torch.topk(out,5,dim=-1)
            total += test_labels.size(0)
            test_labels = test_labels.view(-1,1) 
            top1 += (test_labels == maxk[:,0:1]).sum().item()
            topk += (test_labels == maxk).sum().item()

    return 100 * top1 / total, 100 * topk/total

def evala(model, testdata):

    with torch.no_grad():
        total = 0
        top1 = 0
        topk = 0
        for (test_imgs, test_labels) in tqdm(testdata):

            test_labels = test_labels.cuda()
            _,out = model(test_imgs.cuda())
            _,maxk = torch.topk(out,5,dim=-1)
            total += test_labels.size(0)
            test_labels = test_labels.view(-1,1) 
            top1 += (test_labels == maxk[:,0:1]).sum().item()
            topk += (test_labels == maxk).sum().item()

    return 100 * top1 / total, 100 * topk / total



def lr_cos(step):
        progress = float(step) / float(max(
            1, 50))
        return max(
            0.0,
            0.5 * (1. + math.cos(math.pi * 0.5 * 2.0 * progress))
        )
    
    
def make_optimizer(model,model2=None,model3=None,base_lr=0.002,iround=1,WEIGHT_DECAY=1e-5):
    params = []
    # only include learnable params
    if model2 == None:
        for key, value in model.named_parameters():
            if value.requires_grad and key == 'ctx_local':
                params.append((key, value))

    else:
        for key, value in model.named_parameters():
            if value.requires_grad and key == 'ctx_global':
                params.append((key, value))
        
            
    if model2 != None:
        for key, value in model2.named_parameters():
            if value.requires_grad:
                params.append((key, value))
    
    if model3 != None:
        for key, value in model3.named_parameters():
            if value.requires_grad:
                params.append((key, value))


    _params = []
    for p in params:
        key, value = p
        tlr = base_lr
        weight_decay = WEIGHT_DECAY
        _params += [{
            "params": [value],
            "lr": tlr,
            "weight_decay": weight_decay
        }]

    optimizer = torch.optim.SGD(
        _params,lr = base_lr,momentum=0.9,weight_decay=WEIGHT_DECAY
    )
    return optimizer


class CustomDataset(Dataset):
    def __init__(self, root_dir, phase='train', subfolder=None, transform=None):
        self.root_dir = root_dir
        self.phase = phase  # 'train' or 'test'
        self.transform = transform
        
     
        self.data_dir = os.path.join(root_dir, phase)
        self.subfolder = subfolder
        
  
        self.image_paths = []
        self.labels = []
        self.class_names = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]

 
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}


        subfolder_path = os.path.join(self.data_dir, subfolder)
        for class_name in sorted(os.listdir(subfolder_path)):
            class_path = os.path.join(subfolder_path, class_name)

            for img_name in sorted(os.listdir(class_path)):
                if img_name.endswith('.jpg') or img_name.endswith('.png'):
                    img_path = os.path.join(class_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)

        return img, label


class DomainNetDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                path, label = line.strip().split()
                self.image_paths.append(os.path.join("/data/yubin021/domainnet", path))
                self.labels.append(int(label))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CustomHomeDataset(Dataset):
    def __init__(self, root_dir, phase='train', subfolder=None, transform=None):
        self.root_dir = root_dir
        self.phase = phase  # 'train' 或 'test'
        self.transform = transform
        

        self.data_dir = os.path.join(root_dir, phase)
        self.subfolder = subfolder
        

        self.image_paths = []
        self.labels = []
        self.class_names = ['backpack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']

      
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

     
        subfolder_path = os.path.join(self.data_dir, subfolder)
        for class_name in os.listdir(subfolder_path):
            class_path = os.path.join(subfolder_path, class_name)
        
            for img_name in os.listdir(class_path):
                if img_name.endswith('.jpg') or img_name.endswith('.png'):
                    img_path = os.path.join(class_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)

        return img, label


class CustomDRDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                path, label = line.strip().split()
                self.image_paths.append(os.path.join(txt_file[:-4], path))
                self.labels.append(int(label))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label
