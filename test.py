import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torchinfo import summary

path = "./Dataset/"
mmscaler = MinMaxScaler(feature_range=(-1, 1), copy=True)


#VARIABLE
train_size = 7352
test_size = 2947
all_size = 10299
dim_size = 3
sample_size = 128
batch_size = 1


"""
DATA LOAD
"""
#TRAINING SET 
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

"""
DATA 

X
train : <...128...>  
        x
        y               * 7352
        z

test : <...128...>  
        x
        y               * 2947
        z

Y
subject 人
movement
1,1,1,2,2,2 ....

"""


"""
DATA FORMAT
"""
trY = train_t.reshape(-1,1)
teY = test_t.reshape(-1,1)

teX = np.ones((test_size, dim_size, sample_size), float)
print('teX.shape initial:{0}'.format(teX.shape))
for i in range(test_size):
  teX[i,0,:] = test_ax[i,:]
  teX[i,1,:] = test_ay[i,:]
  teX[i,2,:] = test_az[i,:]

trX = np.ones((train_size, dim_size, sample_size), float)
for i in range(train_size):
  trX[i,0,:] = train_ax[i,:]
  trX[i,1,:] = train_ay[i,:]
  trX[i,2,:] = train_az[i,:]

#array_datax = np.vstack([trX, teX])
#array_datay = np.vstack([trY, teY])

#APPLY FOR TORCH


#dataX = torch.Tensor(np.array(array_datax))
#dataY = torch.Tensor(np.array(array_datay))

trainX = torch.Tensor(np.array(trX))
#trainX = torch.Tensor(train_ax)
#print(np.concatenate([a1, a2]))


trainY = torch.Tensor(np.array(trY))

testX = torch.Tensor(np.array(teX))
testY = torch.Tensor(np.array(teY))

trainX.view(batch_size,train_size,3,sample_size)

class MyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 70
        self.lstm = nn.LSTM(input_size=3, hidden_size=self.hidden_size, num_layers=1, batch_first=True) 
        self.fc = nn.Softmax()

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_size)
        c_0 = torch.zeros(1, x.size(0), self.hidden_size)
        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        y_hat = self.fc(h_out)
        return y_hat

print(summary(MyLSTM(), input_size=(7352, 128, 3), device=torch.device("cpu")))  