import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import copy

from networks import Encoder

class DuplicatedCompose(object):
    """Class for transforming images given a set of transformations
    """
    def __init__(self, transforms):
        """Initalizes image trasnformer object
        transforms: a list of torchvision transforms
        """
        self.transforms = transforms

    def __call__(self, img):
        """Transforms image and returns two copies, one for query and one for queue"""
        img1 = img.copy()
        img2 = img.copy()
        for t in self.transforms:
            img1 = t(img1)
            img2 = t(img2)
        return img1, img2

def momentum_update(model_q, model_k, beta = 0.999):
    param_k = model_k.state_dict() # key encoder params
    param_q = model_q.named_parameters() #  query encoder params
    for n, q in param_q:
        if n in param_k:
            # apply momentum update to key encoder
            param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
    model_k.load_state_dict(param_k)

def queue_data(data, k):
    return torch.cat([data, k], dim=0)

def dequeue_data(data, K=4096):
    if len(data) > K:
        return data[-K:]
    else:
        return data

def initialize_queue(model_k, device, train_loader):
    queue = torch.zeros((0, 128), dtype=torch.float)
    queue = queue.to(device)

    for batch_idx, (data, target) in enumerate(train_loader):
        x_k = data[1]
        x_k = x_k.to(device)
        k = model_k(x_k)
        k = k.detach()
        queue = queue_data(queue, k)
        queue = dequeue_data(queue, K = 10)
        break
    return queue

def train(model_q, model_k, device, train_loader, queue, optimizer, epoch, temp=0.07):
    model_q.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        x_q = data[0]
        x_k = data[1]

        x_q, x_k = x_q.to(device), x_k.to(device)
        q = model_q(x_q)
        k = model_k(x_k)
        k = k.detach()

        N = data[0].shape[0]
        K = queue.shape[0]
        l_pos = torch.bmm(q.view(N,1,-1), k.view(N,-1,1))
        l_neg = torch.mm(q.view(N,-1), queue.T.view(-1,K))

        logits = torch.cat([l_pos.view(N, 1), l_neg], dim=1)

        labels = torch.zeros(N, dtype=torch.long)
        labels = labels.to(device)

        cross_entropy_loss = nn.CrossEntropyLoss()
        loss = cross_entropy_loss(logits/temp, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        momentum_update(model_q, model_k)

        queue = queue_data(queue, k)
        queue = dequeue_data(queue)

    total_loss /= len(train_loader.dataset)

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, total_loss))
