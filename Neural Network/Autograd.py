#==========================================
# Vandit Gajjar
# Date - 27/09/2019
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
from torch import optim

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
optimizer = optim.SGD(model.parameters(), lr = 0.003)

epochs = 5

for e in range(epochs):
	running_loss = 0
	for images, labels in trainloader:

		images = images.view(images.shape[0], -1)
		
		optimizer.zero_grad()
		output = model.forward(images)
		loss = criterion(output, labels)
		loss.backward()
		optimizer.step()
		#print('Gradient -', model[0].weight.grad)
		running_loss += loss.item()
	else:
		print(f"Training loss: {running_loss/len(trainloader)}")

images, labels = next(iter(trainloader))
images = images[0].view(1, 784)
with torch.no_grad():
	logits = model.forward(images)

Prob = F.softmax(logits, dim = 1)
helper.view_classify(images.view(1, 28, 28), Prob)