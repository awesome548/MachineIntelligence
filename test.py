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

train_size = 6866
test_size = 3433
all_size = 10299
dim_size = 3
sample_size = 128
batch_size = 1
"""
==description==
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
data_subject = np.hstack((data_train_subject, data_test_subject))

"""
DATA PREPARATION
"""
#WHOLE SET 
data_ax = np.vstack((data_train_ax, data_test_ax))
data_ay = np.vstack((data_train_ay, data_test_ay))
data_az = np.vstack((data_train_az, data_test_az))

#sub=23 -> 0 , else -> 0 
data_y=np.where(data_subject==23,0,1)
data_y = data_y.reshape(-1,1)
#Whole label size : (10299,1)

test_ax = np.array(data_ax[::3])
test_ay = np.array(data_ay[::3])
test_az = np.array(data_az[::3])
test_Y = np.array(data_y[::3])
train_ax = np.vstack((data_ax[1::3],data_ax[2::3]))
train_ay = np.vstack((data_ay[1::3],data_ay[2::3]))
train_az = np.vstack((data_az[1::3],data_az[2::3]))
train_Y = np.vstack((data_y[1::3],data_y[2::3]))
#test size (3433,128) (3433,1)
#train size (6866,128) (6866,1)
#print(test_ax.shape,test_y.shape,train_ax.shape,train_y.shape)

test_X = np.ones((test_size,sample_size,dim_size), float)
for i in range(test_size):
        test_X[i,:,0] = test_ax[i,:]
        test_X[i,:,1] = test_ay[i,:]
        test_X[i,:,2] = test_az[i,:]

train_X = np.ones((train_size,sample_size,dim_size), float)
for i in range(train_size):
        train_X[i,:,0] = train_ax[i,:]
        train_X[i,:,1] = train_ay[i,:]
        train_X[i,:,2] = train_az[i,:]

data_X = np.vstack([train_X, test_X])
data_Y = np.vstack([train_Y, test_Y])
#Xsize test:(3433, 128, 3) train:(6866, 128, 3) whole:(10299, 128, 3)
#Ysize test:(3433, 1) train:(6866, 1) whole:(10299, 1)
#print(test_X.shape,train_X.shape,data_X.shape)
#print(test_Y.shape,train_Y.shape,data_Y.shape)

"""
APPLY FOR TORCH
"""
dataX = torch.Tensor(np.array(data_X)).to(device)
dataY = torch.Tensor(np.array(data_X)).to(device)
trainX = torch.Tensor(np.array(train_X)).to(device)
trainY = torch.Tensor(np.array(train_Y)).to(device)
testX = torch.Tensor(np.array(test_X)).to(device)
testY = torch.Tensor(np.array(test_Y)).to(device)

#ONE HOT VECTOR
trainY = F.one_hot(trainY.view(-1).long(), num_classes=2)
trainY = (trainY.float()).cpu().data.numpy()
testY = testY.view(-1).long()
