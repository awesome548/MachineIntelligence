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
train_Y = data_train_movement.reshape(-1,1)
test_Y = data_test_movement.reshape(-1,1)

test_X = np.ones((test_size,sample_size,dim_size), float)
print('teX.shape initial:{0}'.format(test_X.shape))
for i in range(test_size):
        test_X[i,:,0] = data_test_ax[i,:]
        test_X[i,:,1] = data_test_ay[i,:]
        test_X[i,:,2] = data_test_az[i,:]

train_X = np.ones((train_size,sample_size,dim_size), float)
for i in range(train_size):
        train_X[i,:,0] = data_train_ax[i,:]
        train_X[i,:,1] = data_train_ay[i,:]
        train_X[i,:,2] = data_train_az[i,:]

datax = np.vstack([train_X, test_X])
datay = np.vstack([train_Y, test_Y])

#APPLY FOR TORCH

dataX = torch.Tensor(np.array(datax)).to(device)
dataY = torch.Tensor(np.array(datay)).to(device)

trainX = torch.Tensor(np.array(train_X)).to(device)
trainY = torch.Tensor(np.array(train_Y)).to(device)

testX = torch.Tensor(np.array(test_X)).to(device)
testY = torch.Tensor(np.array(test_Y)).to(device)

#ONE HOT VECTOR 

trainY = trainY.view(-1).long() - 1
trainY = F.one_hot(trainY, num_classes=-1)
trainY = trainY.float()
trainY = trainY.cpu().data.numpy()
testY = testY.view(-1).long() - 1
# testY = F.one_hot(testY, num_classes=-1)
# testY = testY.float()

"""
CLASS DEFINITION
"""

class MyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size_1 = 50
        self.hidden_size_2 = 70
        self.lstm_1 = nn.LSTM(input_size=3, hidden_size=self.hidden_size_1, num_layers=1, batch_first=True) 
        self.lstm_2 = nn.LSTM(input_size=self.hidden_size_1, hidden_size=self.hidden_size_2, num_layers=1, batch_first=True) 
        self.relu = nn.ReLU()
        self.linear = nn.Linear(70, 6)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_size_1).to(device)
        c_0 = torch.zeros(1, x.size(0), self.hidden_size_1).to(device)
        h_1 = torch.zeros(1, x.size(0), self.hidden_size_2).to(device)
        c_1 = torch.zeros(1, x.size(0), self.hidden_size_2).to(device)
        out, (h_out, c_out) = self.lstm_1(x, (h_0, c_0))
        _, (h_out, _) = self.lstm_2(out, (h_1, c_1))
        h_out = h_out.view(-1, self.hidden_size_2)
        h_out = self.relu(h_out)
        y_hat = self.linear(h_out)
        return y_hat


class DataSet:
        def __init__(self):
                self.X = train_X.astype(np.float32) 
                self.t = trainY
        def __len__(self):
                return len(self.X)
        def __getitem__(self, index):
                return self.X[index], self.t[index]

dataset = DataSet()
dataloader = torch.utils.data.DataLoader(dataset,batch_size=128,shuffle=True, drop_last=True)

def train(model, optimizer, X, t):
  model.train()
  y_hat = model(X)
  loss = nn.CrossEntropyLoss()
  output = loss(y_hat, t)
  optimizer.zero_grad()
  output.backward()
  optimizer.step()
  return output.item()

loss = []

def main():
        model = MyLSTM()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model = model.to(device)
        for epoch in range(200):
                for X, t in dataloader:
                        _loss = train(model, optimizer, X.to(device), t.to(device))
                        loss.append(_loss)
                if epoch % 20 == 0:
                        print(f"Epoch = {epoch+1}, Loss = {_loss:.5f}")
        return model

model = main()

def predict(model):
        model.eval()
        train_predict = model(testX)
        
        data_predict = torch.argmax(train_predict, dim=-1)
        data_predict = F.one_hot(data_predict, num_classes=-1)

predict(model)

