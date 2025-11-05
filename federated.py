import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
import numpy
import copy
import random
from tqdm import tqdm
import numpy as np
import argparse
import logging
import timm
import math
import utils
import logging
from setting import get_parser
from clip_etf import CustomCLIP_client
import clip
from utils import make_optimizer,evala,evalaa
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from PIL import Image


_tokenizer = _Tokenizer()

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


parser = get_parser()
args = parser.parse_args()

def setup_seed(seed):  # setting up the random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(args.seed)


logging.basicConfig(
    filename="experiment.log",  # 日志文件路径
    filemode="a",  # 追加模式，不覆盖已有日志
    format="%(asctime)s - [Round %(round)d] [Client %(client)d] - %(message)s",
    level=logging.INFO  # 记录 INFO 级别日志
)
logger = logging.getLogger()

def generate_etf_basis(d, C):
    W = torch.randn(d, C)
    W, _ = torch.qr(W)
    H = torch.eye(C) - (1 / C) * torch.ones((C, C))
    W = W @ H
    W = W / W.norm(dim=0, keepdim=True)

    return W   
    
model, preprocess = clip.load("ViT-B/32", device=args.device)


#======================= dataset ==========================================
if args.data == 'office':
    numclass=10
    domains = ['amazon', 'caltech', 'dslr', 'webcam']
    client_testloaders,client_dataloaders,client_datasets = [],[],[]
    for domain in domains:
        train_data =  utils.CustomHomeDataset(args.data_path, "train", domain, transform=data_transform)
        test_data =  utils.CustomHomeDataset(args.data_path, "test", domain, transform=test_transform)
        client_dataloaders.append(torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=8))
        client_testloaders.append(torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=8))
    
    out = ['backpack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']


if args.data == 'pacs':
    numclass=7
    domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    client_testloaders,client_dataloaders,client_datasets = [],[],[]
    count = 0
    for domain in domains:
        train_data =  utils.CustomDataset(args.data_path, "train", domain, transform=data_transform)
        test_data =  utils.CustomDataset(args.data_path, "test", domain, transform=test_transform)
        client_dataloaders.append(torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=8))
        client_testloaders.append(torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=8))
        count += 1
    
    out = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]


if args.data == 'DR':
    numclass=6
    domains = ['domain_A', 'domain_B', 'domain_C', 'domain_D']
    client_testloaders,client_dataloaders,client_datasets = [],[],[]
    for domain in domains:
        train_data =  utils.CustomDRDataset(args.data_path + domain + "/train.txt", transform=preprocess)
        test_data =  utils.CustomDRDataset(args.data_path + domain + "/test.txt", transform=preprocess)
        client_dataloaders.append(torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=8))
        client_testloaders.append(torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=8))
    
    out = ["No_Diabetic_Retinopathy", "Mild_Non-Proliferative_Diabetic_Retinopathy", "Moderate_Non-Proliferative_Diabetic_Retinopathy", 
            "Severe_Non-Proliferative_Diabetic_Retinopathy", "Proliferative_Diabetic_Retinopathy", "Unreadable_Images_For_Diagnosis"]

if args.data == 'domain':
    numclass=126
    domains = ['clipart', 'painting', 'real', 'sketch']
    client_testloaders,client_dataloaders,client_datasets = [],[],[]
    for domain in domains:
        train_data = utils.DomainNetDataset(args.data_path + domain + "/" + domain +  "_126_train.txt", transform=preprocess)
        test_data = utils.DomainNetDataset(args.data_path + domain + "/" + domain +  "_126_test.txt", transform=preprocess)
        client_dataloaders.append(torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=8))
        client_testloaders.append(torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=8))
    

    class_names = [None for i in range(126)]

    with open(args.data_path + "clipart_126.txt", "r") as fp:
        for line in fp.readlines():
            content = line.strip().split(" ")[0]
            label = line.strip().split(" ")[1]
            label = int(label)
            category = os.path.basename(os.path.dirname(content))
            category = category.replace("_", " ")
            class_names[label] = category
    
    out = class_names




#=======================     Initialize keys and models  ==========================================
global_model = CustomCLIP_client(out, model,args.gctx, domain_number=len(domains)).to(args.device)
models  = [CustomCLIP_client(out, model,args.gctx, domain_number=len(domains)).to(args.device) for i in range(len(client_dataloaders))]
for client in models[1:]:
    client.load_state_dict(models[0].state_dict())

# generate the semantic ETF and domain ETF
sem_etf = generate_etf_basis(512, numclass).half().to(args.device)
dom_etf = generate_etf_basis(512, len(domains)).half().to(args.device)
print(numclass, len(domains))

# model initialization
for model in models:
    for name, param in model.named_parameters():
        if "prompt_learner" not in name and "sem_trans" not in name and "dom_trans" not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

#=======================     Federated training  ==========================================
for fe in range(args.round):
    # logger.info(f'------------- federated {fe}-th  --------------------------')
    print('------------- federated ',fe,'-th  --------------------------')
        
    for cl,client in enumerate(tqdm(models)):
        optimizer1 = make_optimizer(client.prompt_learner, client.sem_trans, client.dom_trans, base_lr=args.lr)
        optimizer2 = make_optimizer(client.prompt_learner, base_lr=args.lr)

        for e in range(1):
            for i, (image, label) in enumerate(tqdm(client_dataloaders[cl])):
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                image = image.to(args.device)
                label = label.to(args.device)
                # print(label)

                out_cls, img_sem_log, img_dom_log, txt_sem_log, txt_dom_log = client(image)
                # contrastive loss
                contras_loss = F.cross_entropy(out_cls, label)

                img_sem_log = img_sem_log / img_sem_log.norm(dim=-1, keepdim=True)
                img_sem_log = img_sem_log @ sem_etf
                # loss for semantic TN
                sem_TN_loss = F.cross_entropy(img_sem_log, label)

                dom_label = torch.tensor([cl for i in range(label.shape[0])]).to(args.device)
                img_dom_log = img_dom_log / img_dom_log.norm(dim=-1, keepdim=True)
                img_dom_log = img_dom_log @ dom_etf
                # loss for domain TN
                dom_TN_loss = F.cross_entropy(img_dom_log, dom_label)

                txt_sem_log = txt_sem_log / txt_sem_log.norm(dim=-1, keepdim=True)
                txt_sem_log = txt_sem_log @ sem_etf
                sem_label = torch.tensor([i for i in range(len(out))]).to(args.device)
                # loss for the alignment of global semantic prompt
                sem_align_loss = F.cross_entropy(txt_sem_log, sem_label)

                txt_dom_log = txt_dom_log / txt_dom_log.norm(dim=-1, keepdim=True)
                txt_dom_log = txt_dom_log @ dom_etf
                dom_label = torch.tensor([cl for i in range(len(out))]).to(args.device)
                # loss for the alignment of personalied local prompt
                dom_align_loss = F.cross_entropy(txt_dom_log, dom_label)

                # loss for the optimization of global prompt
                glb_loss = contras_loss + sem_TN_loss + dom_TN_loss + args.lamda * sem_align_loss

                # loss for the optimization of local prompt
                loc_loss = contras_loss + args.sigma * dom_align_loss
                
                # keep the gradient of contrastive loss
                glb_loss.backward(retain_graph=True)
                optimizer1.step()

                loc_loss.backward()
                optimizer2.step()

            
    weights = [1/len(models)]*len(models)

    prompt = [] 
    prompt_state, local_sem_state, local_dom_state = models[0].prompt_learner.state_dict(), models[0].sem_trans.state_dict(), models[0].dom_trans.state_dict()

    # parameter aggregation
    for k, client in enumerate(models):
        client_prompt_state = client.prompt_learner.state_dict()
        for st in prompt_state:
            if k==0:
                prompt_state[st] = client_prompt_state[st]*weights[k]
            else:
                prompt_state[st] += client_prompt_state[st]*weights[k]
    
    for k, client in enumerate(models):
        client_sem_state = client.sem_trans.state_dict()
        for st in local_sem_state:
            if k==0:
                local_sem_state[st] = client_sem_state[st]*weights[k]
            else:
                local_sem_state[st] += client_sem_state[st]*weights[k]
    
    for k, client in enumerate(models):
        client_dom_state = client.dom_trans.state_dict()
        for st in local_dom_state:
            if k== 0:
                local_dom_state[st] = client_dom_state[st]*weights[k]
            else:
                local_dom_state[st] += client_dom_state[st]*weights[k]


    global_model.sem_trans.load_state_dict(local_sem_state, strict=False)
    global_model.dom_trans.load_state_dict(local_dom_state, strict=False)

    # update the parameter of global prompt, semantic TN and domain TN
    for m,client in enumerate(models):
        client.prompt_learner.ctx_global.data = prompt_state['ctx_global']
        client.sem_trans.load_state_dict(global_model.sem_trans.state_dict(), strict=False)
        client.dom_trans.load_state_dict(global_model.dom_trans.state_dict(), strict=False)
        

    # test the updated model on each client's test dataset
    for te,test_loader in enumerate(client_testloaders):
        top1,topk = evalaa(models[te],test_loader)
        print('round '+str(fe)+' in client '+str(te)+' acc: ',top1)  
        logger.info(
            f"Round {fe} - Client {te}: Top-1 Accuracy = {top1:.2f}%",
            extra={"round": fe, "client": te}
        )
