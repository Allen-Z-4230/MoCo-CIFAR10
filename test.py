import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from networks import Encoder
from train import get_loader

def test_conv(model, subsample=False):
    """Measures testing accuracy of the baseline
    """
    # load in data
    testloader = get_loader(10, train=False, subsample=subsample)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct/total

def plot_tsne(targets, ret):

    # Load data
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # target indices
    target_ids = range(len(set(targets)))

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'violet', 'orange', 'purple']

    plt.figure(figsize=(12, 10))

    ax = plt.subplot(aspect='equal')
    for label in set(targets):
        idx = np.where(np.array(targets) == label)[0]
        plt.scatter(ret[idx, 0], ret[idx, 1], c=colors[label], label=label)

    for i in range(0, len(targets), 250):
        img = (dataset[i][0] * 0.3081 + 0.1307).numpy()[0]
        img = OffsetImage(img, cmap = 'gray',zoom=1)
        ax.add_artist(AnnotationBbox(img, ret[i]))

    plt.savefig('./results/tsne.png')
    plt.legend()
    plt.show()

def encode_data(train = False, subsample = False):
    """Encodes the dataset with the pretrained unsupervsied encoders
    using either the training or testing data.
    """
    # load model
    model = Encoder()
    model.load_state_dict(torch.load('./results/encoder.pth', map_location=torch.device('cuda')))

    # Load data
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)

    if subsample:  # randomly samples 1/10 of the train/test data for multiple runs
        bound = 50000 if train else 10000
        size = 5000 if train else 1000
        dataset = Subset(dataset, np.random.randint(0, bound, size))

    # encode data using encoder
    data = []
    targets = []
    for m in tqdm.tqdm(dataset):
        target = m[1]
        targets.append(target)
        x = m[0]
        x = x.view(1, *x.shape)
        feat = model(x)
        data.append(feat.data.numpy()[0])

    return np.array(data),  np.array(targets)
