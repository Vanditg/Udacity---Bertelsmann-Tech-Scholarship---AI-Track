#==========================================
# Vandit Gajjar
# Date - 21/09/2019
# Udacity - Bertelsmann Scholarship Content
#==========================================

import torch

#Defining Softmax function
def activation(x):
	return 1/(1 + torch.exp(-x))

torch.manual_seed(7)

#Generating random features (input value) - normal distribution
features = torch.randn((1, 3))
print("features =", features)
print(features.shape)

#Data given for Number of Input, hidden layers, and output. 
n_input = features.shape[1]
n_hidden = 2
n_output = 1

#Generating random weights based on the data
w1 = torch.randn(n_input, n_hidden)
print(w1.shape)
w2 = torch.randn(n_hidden, n_output)
print(w2.shape)

#Bias values 
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

#Solution for multilayer NN
output = activation(torch.mm(features, w1) + B1)
print("Output =", output)
print(output.shape)

new_output = activation(torch.mm(output, w2) + B2)
print("New_output =", new_output)
print(new_output.shape)