import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torchinfo import summary



#VARIABLE
path = "./Dataset/"
mmscaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
device = torch.device('cpu')

train_size = 7352
test_size = 2947
all_size = 10299
dim_size = 3
sample_size = 128
batch_size = 1
"""
dataX,dataY : whole

trainX,trainY : train 

testX,testY : test
"""


"""
DATA LOAD
"""
#TRAINING SET 
data_train_ax = mmscaler.fit_transform(np.loadtxt(path + "train/Inertial Signals/total_acc_x_train.txt"))
data_train_ay = mmscaler.fit_transform(np.loadtxt(path + "train/Inertial Signals/total_acc_y_train.txt"))
data_train_az = mmscaler.fit_transform(np.loadtxt(path + "train/Inertial Signals/total_acc_z_train.txt"))

data_train_movement = np.loadtxt(path + "train/y_train.txt")
data_train_subject = np.loadtxt(path + "train/subject_train.txt")

#test set
data_test_ax = mmscaler.fit_transform(np.loadtxt(path + "test/Inertial Signals/total_acc_x_test.txt"))
data_test_ay = mmscaler.fit_transform(np.loadtxt(path + "test/Inertial Signals/total_acc_y_test.txt"))
data_test_az = mmscaler.fit_transform(np.loadtxt(path + "test/Inertial Signals/total_acc_z_test.txt"))


data_test_movement = np.loadtxt(path + "test/y_test.txt")
data_test_subject = np.loadtxt(path + "test/subject_test.txt")

data_ax = np.vstack((data_train_ax, data_test_ax))
data_ay = np.vstack((data_train_ay, data_test_ay))
data_az = np.vstack((data_train_az, data_test_az))
data_subject = np.hstack((data_train_subject, data_test_subject))

print(data_test_subject.shape,data_train_subject.shape)

data_y=(data_subject==23)

print(data_y)