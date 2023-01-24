import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import argparse
import tensorboardX
import os
import random
import numpy as np
from utils import AverageMeter, calculate_accuracy

def train_epoch(model, data_loader, criterion, optimizer, epoch, log_interval, device):
    model.train()
 
    train_loss = 0.0
    losses = AverageMeter()
    accuracies = AverageMeter()
    for batch_idx, (data, targets) in enumerate(data_loader):
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        """ 
        print(f'[train.py] 正在执行train_epoch函数，targets.shape={targets.shape}, 输入的data的shape为{data.shape}，输出的outputs的shape为{outputs.shape}')
        targets.shape=torch.Size([8]), 输入的data的shape为torch.Size([8, 16, 3, 150, 150])，输出的outputs的shape为torch.Size([8, 2])
        # print(f'[train.py] 正在执行train_epoch函数，model={model}')
        # print(f'[train.py] 正在执行train_epoch函数，输入数据data={data}, 真实值targets={targets}, 模型输出outputs={outputs}')
        真实值targets=tensor([0, 0, 0, 0, 1, 0, 0, 0]), 
        模型输出outputs=tensor([[0.0559, 0.0073],
                [0.0578, 0.0087],
                [0.0573, 0.0068],
                [0.0567, 0.0075],
                [0.0573, 0.0096],
                [0.0559, 0.0073],
                [0.0552, 0.0075],
                [0.0549, 0.0070]]
         """

        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        train_loss += loss.item()
        losses.update(loss.item(), data.size(0))
        accuracies.update(acc, data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = train_loss / log_interval
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(data_loader.dataset), 100. * (batch_idx + 1) / len(data_loader), avg_loss))
            train_loss = 0.0

    print('Train set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
        len(data_loader.dataset), losses.avg, accuracies.avg * 100))

    return losses.avg, accuracies.avg  