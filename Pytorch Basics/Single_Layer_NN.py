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
features = torch.randn((1, 5))
print("features =", features)
print(features.shape)

#Generating random weights - Same shape as features
weights = torch.randn_like(features)
print("weights =", weights)
print(weights.shape)
#Reshaping the weights current shape
new_weights = weights.reshape(5, 1)
print(new_weights.shape)

bias = torch.rand((1, 1))
print("bias =", bias)
print(bias.shape)

#Caluclating the output using torch.sum
output = activation(torch.sum(features*weights) + bias)
print("output =", output)
print(output.shape)

#Matrix multiplication
matrixMul = torch.mm(features, new_weights)
print("matrixMul =", matrixMul)
print(matrixMul.shape)

#Caluclating the output using torch.mm - Matrix Multiplication
new_output = activation(matrixMul + bias)
print("new_output =", new_output)
print(new_output.shape)