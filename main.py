from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torchvision import torch
import torch.nn as nn
import torch.optim as optim
import glob
from Functions import *
from Classifier import *
from TrainTestModel import *
import cifar

def main():
    cifar.init()
    train_data, test_data = loadDataSet()
    train_loader, valid_loader, test_loader = loader(train_data, test_data)

    visualize(train_loader)

    model = Net()
    print(model)

    # Specify loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    if (glob.glob('model_cifar.pt')):
        model.load_state_dict(torch.load('model_cifar.pt'))
        test(model, test_loader, criterion)
    else:
        train(model, train_loader, valid_loader, criterion, optimizer)

main()