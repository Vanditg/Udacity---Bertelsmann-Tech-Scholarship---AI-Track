#==========================================
# Vandit Gajjar
# Date - 21/09/2019
# Udacity - Bertelsmann Scholarship Content
#==========================================

import numpy as np
import torch
import helper
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

#Downloading MNIST data	
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

#Defining input, hidden and output units for network 
number_input = 784
number_hidden_1 = 128
number_hidden_2 = 64
number_output = 10

#Building sequential model using logsoftmax and setting criterion to NLL loss. 
model = nn.Sequential(nn.Linear(number_input, number_hidden_1), 
                      nn.ReLU(), 
                      nn.Linear(number_hidden_1, number_hidden_2),
                      nn.ReLU(), 
                      nn.Linear(number_hidden_2, number_output), 
					  nn.LogSoftmax(dim = 1))

criterion = nn.NLLLoss()

images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logits = model(images)
loss = criterion(logits, labels)
print("Loss =", loss)