import tensorflow as tensor
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from src.data.make_dataset import DataCorpusEncoderManager, DataSplitter

# Initialize the data splitter and encoder manager from make_dataset.py
ds = DataSplitter()
dc = DataCorpusEncoderManager()

# Relative path from C:\Users\summe\OneDrive\Documents\New-Projects\undectable_ai\src\models
x_train, y_train, x_test, y_test = ds.split_dataset(
    '../../project_data/processed/QuestionAnswers2.csv', 
    ['Response'], 
    'Ai-generated'
)

x_train_encoded = dc.encode_corpus(x_train)

print(x_train_encoded)