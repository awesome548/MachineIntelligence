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

#training set
train_ax = mmscaler.fit_transform(np.loadtxt(path + "train/Inertial Signals/total_acc_x_train.txt"))
train_ay = mmscaler.fit_transform(np.loadtxt(path + "train/Inertial Signals/total_acc_y_train.txt"))
train_az = mmscaler.fit_transform(np.loadtxt(path + "train/Inertial Signals/total_acc_z_train.txt"))

train_t = np.loadtxt(path + "train/y_train.txt")
train_s = np.loadtxt(path + "train/subject_train.txt")

#test set
test_ax = mmscaler.fit_transform(np.loadtxt(path + "test/Inertial Signals/total_acc_x_test.txt"))
test_ay = mmscaler.fit_transform(np.loadtxt(path + "test/Inertial Signals/total_acc_y_test.txt"))
test_az = mmscaler.fit_transform(np.loadtxt(path + "test/Inertial Signals/total_acc_z_test.txt"))

test_t = np.loadtxt(path + "test/y_test.txt")
test_s = np.loadtxt(path + "test/subject_test.txt")