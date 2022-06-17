import csv

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
#from torchinfo import summary


path = "./Dataset/"
mmscaler = MinMaxScaler(feature_range=(-1, 1), copy=True)

ax = mmscaler.fit_transform(np.loadtxt(path + "train/Inertial Signals/total_acc_x_train.txt"))
ay = mmscaler.fit_transform(np.loadtxt(path + "train/Inertial Signals/total_acc_y_train.txt"))
az = mmscaler.fit_transform(np.loadtxt(path + "train/Inertial Signals/total_acc_z_train.txt"))

t = np.loadtxt(path + "train/y_train.txt")
s = np.loadtxt(path + "train/subject_train.txt")

print(ax)
print(t)
print(s)