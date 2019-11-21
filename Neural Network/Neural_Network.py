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
from torchvision import datasets, transforms

#Defining Sigmoid 
def activation(x):
	return 1/(1 + torch.exp(-x))

#Defining Softmax	
def softmax(x):
	return torch.exp(x)/torch.sum(torch.exp(x), dim = 1).reshape(-1, 1)

#Downloading MNIST data	
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

#Showing one of the image
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')

#Defining input, hidden and output units for network 
number_input = images.view(images.shape[0], -1)
number_hidden = 256
number_output = 10

#Generating Weight and Bias metrices - W1, W2, B1, and B2.
W1 = torch.randn(784, number_hidden)
print("W1 =", W1.shape)
W2 = torch.randn(number_hidden, number_output)
print("W2 =", W2.shape)

B1 = torch.randn(number_hidden)
B2 = torch.randn(number_output)

#Calculating output from Input, W1. 
output = activation(torch.mm(number_input, W1) + B1)
print("Output =", output.shape)

#Calculating output for output, W2. 
new_output = torch.mm(output, W2) + B2
print("New Output =", new_output.shape)

#Calculating probabilities from our softmax function. 
prob = (softmax(new_output))
print("prob = ", prob)
print("prob shape =", prob.shape)
print("Total sum =", prob.sum(dim = 1))