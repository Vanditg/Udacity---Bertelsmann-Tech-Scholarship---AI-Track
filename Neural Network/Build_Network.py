#==========================================
# Vandit Gajjar
# Date - 21/09/2019
# Udacity - Bertelsmann Scholarship Content
#==========================================

import numpy as np
import helper
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import nn

number_input = 784
number_hidden_1 = 128
number_hidden_2 = 64
number_output = 10

class NN(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.output = nn.Linear(number_input, number_hidden_1)
		self.new_output = nn.Linear(number_hidden_1, number_hidden_2)
		self.final_output = nn.Linear(number_hidden_2, number_output)
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax(dim = 1)
	
	def forward(self, x):
		x = self.output(x)
		x = self.relu(x)
		x = self.new_output(x)
		x = self.relu(x)
		x = self.final_output(x)
		x = self.softmax(x)
		
		return x

model = NN()
print(model)

class New_NN(nn.Module):
	def __init__(self):
		super().__init__()
		self.output = nn.Linear(number_input, number_hidden_1)
		self.new_output = nn.Linear(number_hidden_1, number_hidden_2)
		self.final_output = nn.Linear(number_hidden_2, number_output)
	
	def forward(self, x):
		x = self.output(x)
		x = F.ReLU(self.output(x))
		x = self.new_output(x)
		x = F.ReLU(self.new_output(x))
		x = self.final_output(x)
		x = F.softmax(self.final_output(x), dim = 1)
		return x
		
model_new = New_NN()
print(model_new)