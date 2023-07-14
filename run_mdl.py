"""
Author: Zhibin Li: lzb5600@gmail.com
"""
import itertools
import random
import torch
import tqdm
import copy
import argparse
import numpy as np
import sys
from time import time
from mdl import corr_meta_learning
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from gen_dataset.avazu import AvazuDataset
from gen_dataset.criteo import CriteoDataset
from gen_dataset.mltag import MLDataset
from gen_dataset.frappe import FrappeDataset
from gen_dataset.kdd2012 import KDD2012Dataset, split_index


def get_dataset(path):
    if 'criteo' in path:
        return CriteoDataset(path)
    elif 'avazu' in path:
        return AvazuDataset(path)
    elif 'frappe' in path:
        return FrappeDataset(path)
    elif 'adult' in path:
        return AdultDataset(path)
    elif 'mltag' in path:
        return MLDataset(path)
    elif 'kdd2012' in path:
        return KDD2012Dataset(path)
    else:
        raise ValueError('unknown dataset name')


def train(model, optimizer, data_loader, criterion, device):
    model.train()
    for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, file=sys.stdout, disable=False):#True):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.w.data = model.projection_w(model.w.data.unsqueeze(0)).squeeze(0)


def test(model, data_loader, criterion, device):
    model.eval()
    targets, predicts = list(), list()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for fields, target in data_loader:
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            loss = criterion(y, target.float())
            total_loss += loss.item()*len(y)
            total_samples += len(y)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts),total_loss/total_samples


def main(args):
    device = torch.device(args.device)
    dataset = get_dataset(args.dataset_path)
    if "kdd2012" in args.dataset_path: #for getting the same dataset split with another paper
        train_idx,val_idx,test_idx = split_index(len(dataset))
    else:
        train_length = int(len(dataset) * 0.8)
        valid_length = int(len(dataset) * 0.1)
        all_index = list(range(len(dataset)))      
        random.seed(6)
        random.shuffle(all_index)
        train_idx = all_index[:train_length]
        val_idx = all_index[train_length:(train_length + valid_length)]
        test_idx = all_index[(train_length + valid_length):]
    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)    
    model = corr_meta_learning(dataset.field_dims, args.ebd_dim, args.inner_step, args.inner_step_size, args.lmbd, args.layer)
    model.to(device)
    #=============count number of parameters==================
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)) 
    #=========================================================
    criterion = torch.nn.BCEWithLogitsLoss()    
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wdcy)                          

    best_loss = 1e10  
    test_auc = 0
    test_loss = 1e10
    stop_cnt = 0  
    best_epoch = 0
    for epoch_i in range(args.epoch):
        train(model, optimizer, train_data_loader, criterion, device)
        if args.train_info:
            auc,loss = test(model, train_data_loader, criterion, device)
            print("train_auc:",auc,"train_logloss:",loss)
        auc,loss = test(model, valid_data_loader, criterion, device)
        print("valid_auc:",auc,"valid_logloss:",loss)
        if loss<best_loss:
            best_loss = loss
            best_auc = auc
            best_epoch = epoch_i+1
            test_auc,test_loss = test(model, test_data_loader, criterion, device)
            stop_cnt = 0   
        elif stop_cnt<args.stop_window:
            stop_cnt+=1 
        else:
            break
    return(best_epoch, test_auc, test_loss)        



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train and test the model")
    parser.add_argument('--train-info', type=int, default=1, help="display training info")
    parser.add_argument('--dataset-path', type=str, default=None, help="path to the dataset")
    
    parser.add_argument('--inner-step', type=int, default=1, help="number of steps for embedding adjustment (inner-loop of MDL)")
    parser.add_argument('--inner-step-size', type=float, default=1e-3, help="learning rate of inner-loop of MDL")
    parser.add_argument('--lmbd', type=float, default=1, help="lambda, controlling the sparsity of mu")
    parser.add_argument('--ebd-dim', type=int, default=10, help="embedding dimension of categorical features")
    parser.add_argument('--layer', type=int, default=3, help="mlp layers of the backbone")

    parser.add_argument('--lr', type=float, default=5e-2, help="learning rate of outer-loop of MDL")
    parser.add_argument('--wdcy', type=float, default=1e-8, help="weight decay")
    parser.add_argument('--stop-window', type=int, default=0, help="early-stop window")

    parser.add_argument('--batch-size', type=int, default=2048, help="batch size")
    parser.add_argument('--epoch', type=int, default=15, help="max training epochs")
    parser.add_argument('--device', type=str, default="cuda:0", help="device to use")      
    parser.add_argument('--num-workers', type=int, default=1)
             
    args = parser.parse_args()
    a = time()
    test_results = main(args)
    print("epoch: {0}, best_auc: {1}, best_loss: {2}".format(test_results[0],test_results[1],test_results[2]))
    print("time:",time()-a) 
    
