#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 20:10:05 2018

Convolution Neural Network 

@author: mohak
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms

train_dataset = dsets.MNIST(root = '/media/mohak/Work/Datasets/MNIST',
                            train = True,
                            transform = transforms.ToTensor(),
                            download=False)


test_dataset = dsets.MNIST(root = '/media/mohak/Work/Datasets/MNIST',
                           train=False,
                           transform=transforms.ToTensor())

bs = 100
iters = 3000
epochs = 5
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = bs, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = bs, shuffle = False)

class CNN(nn.Module):
    #conv
    #maxpool
    #conv
    #maxpool
    #flatten
    #feedforward
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fcl = nn.Linear(32*7*7, 10)
        
    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fcl(out)
        return out
        
model = CNN().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

iter=0
for epoc in range(epochs):
    for i, (image, label) in enumerate(train_loader):
        image = Variable(image.cuda())
        label = Variable(label.cuda())
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        iter+=1
        if iter%500==0:
            correct = 0
            total = 0
            for images,lables in test_loader:
                images = Variable(images).cuda()
                outputs = model(images)
                _,pred = torch.max(outputs.data,1)
                total += lables.size(0)
                correct += (pred.cpu()==lables.cpu()).sum()
            accuracy = 100*correct/total
            print('iter = {}, acc = {}, loss = {}'.format(iter, accuracy, loss.data[0]))
            

torch.save(model.state_dict,'/media/mohak/Work/Projects/MNIST_CNN_acc97.pkl')
                
