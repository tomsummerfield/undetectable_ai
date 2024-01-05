import sys
import datetime
import tensorflow as tensor
import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
import os.path
import sys
import os
from src.data.make_dataset import DataCorpusEncoderManager, DataSplitter

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )

# print(f"Using {device} device")

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(64, 1),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits
    
# net = NeuralNetwork()

# # Training for the model
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# num_epochs = 100

# # Split dataset
# ds = DataSplitter()
# x_train, y_train, X_test, Y_Test = ds.split_dataset('./data/processed/QuestionAnswers2.csv', ['Response'], 'Ai-generated')

# # Encode the data
# dc = DataCorpusEncoderManager()
# x_train = dc.encode_corpus(x_train)

# # set up DataLoader for training set
# loader = DataLoader(list(zip(x_train, y_train)), shuffle=True, batch_size=16)


# for epoch in range(num_epochs):  # loop over the dataset multiple times

#     running_loss = 0.0

#     for i, data in enumerate(loader, 1):  # Start index from 1

#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()

#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss = 0.0

# print('Finished Training')

# file_path = './src/models/nn1_model'
# torch.save(net.state_dict(), file_path)