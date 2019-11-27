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
trainset = datasets.FashionMNIST('Fashio_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

#Defining input, hidden and output units for network 
number_input = 784
number_hidden_1 = 256
number_hidden_2 = 128
number_hidden_3 = 64
number_hidden_4 = 32
number_output = 10

class NN(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.output = nn.Linear(number_input, number_hidden_1)
		self.new_output = nn.Linear(number_hidden_1, number_hidden_2)
		self.new_final_output = nn.Linear(number_hidden_2, number_hidden_3)
		self.new_final_last_output = nn.Linear(number_hidden_3, number_hidden_4)
		self.new_final_last_fashion_output = nn.Linear(number_hidden_4, number_output)
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax(dim = 1)
	
	def forward(self, x):
		x = self.output(x)
		x = self.relu(x)
		x = self.new_output(x)
		x = self.relu(x)
		x = self.new_final_output(x)
		x = self.relu(x)
		x = self.new_final_last_output(x)
		x = self.relu(x)
		x = self.new_final_last_fashion_output(x)
		x = self.softmax(x)
		
		return x

model = NN()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.003)

epochs = 25

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