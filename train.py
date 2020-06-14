import os
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset
from PIL import Image
import numpy as np
from dataclasses import dataclass
import copy

from networks import Encoder, ConvNet

@dataclass
class TrainArgs:
    """Settings for training"""
    batchsize: int = 100
    epochs: int = 10
    out: str = './results'
    no_cuda: bool = False

def get_loader(batchsize, train=True, subsample = False):
    """Gets a loader of un-augmented images for testing / baseline training"""
    kwargs = {'num_workers': 8, 'pin_memory': True, 'shuffle':True}

    # datasets
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                        download=True, transform=transform)
    if subsample:  # randomly samples 1/10 of the train/test data for multiple runs
        bound = 50000 if train else 10000
        size = 5000 if train else 1000
        trainset = Subset(trainset, np.random.randint(0, bound, size))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, **kwargs)
    return trainloader

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

    return total_loss

def train_encoder(args):
    """Trains an encoder using given settings with MoCo.

    args: custom dataclass of training/model settings
    """

    #settings
    batchsize = args.batchsize
    epochs = args.epochs
    out_dir = args.out

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 8, 'pin_memory': True}

    # a bunch of random image augmentations
    transform = DuplicatedCompose([
        transforms.RandomResizedCrop(32, scale=(0.9, 1.1), ratio=(0.9, 1.1)),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # train & test datasets
    train_cf10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_cf10 = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_cf10, batch_size=batchsize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_cf10, batch_size=batchsize, shuffle=True, **kwargs)

    # initialize encoders and queue
    model_q = Encoder().to(device)
    model_k = copy.deepcopy(model_q)
    optimizer = optim.SGD(model_q.parameters(), lr=0.01, weight_decay=0.0001)

    queue = initialize_queue(model_k, device, train_loader)

    # start training
    loss_curve = []
    for epoch in range(1, epochs + 1):
        loss_curve.append(train(model_q, model_k, device, train_loader, queue, optimizer, epoch))

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model_q.state_dict(), os.path.join(out_dir, 'encoder.pth'))

    return model_q, loss_curve

def train_ConvNet(args, subsample=False):
    """Trains a standard convolutional network with given parameters
    """

    #settings
    batchsize = args.batchsize
    epochs = args.epochs
    out_dir = args.out
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader = get_loader(batchsize, subsample=subsample)

    #initiate network
    net = ConvNet()
    net.to(device)

    # We use cross-entropy as loss function.
    loss_func = nn.CrossEntropyLoss()
    # We use stochastic gradient descent (SGD) as optimizer.
    opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    avg_losses = []   # Avg. losses.
    print_freq = 500  # Print frequency.

    # looping across epochs
    print('Started looping')
    for epoch in range(epochs):  # Loop over the dataset multiple times.
        running_loss = 0.0       # Initialize running loss.
        for i, data in enumerate(trainloader, 0):
            # Get the inputs.
            inputs, labels = data

            # Move the inputs to the specified device.
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients.
            opt.zero_grad()

            # Forward step.
            outputs = net(inputs)
            loss = loss_func(outputs, labels)

            # Backward step.
            loss.backward()

            # Optimization step (update the parameters).
            opt.step()

            # Print statistics.
            running_loss += loss.item()
            if i % print_freq == print_freq - 1: # Print every several mini-batches.
                avg_loss = running_loss / print_freq
                print('[epoch: {}, i: {:5d}] avg mini-batch loss: {:.3f}'.format(
                    epoch, i, avg_loss))
                avg_losses.append(avg_loss)
                running_loss = 0.0

    return net, np.array(avg_losses)
