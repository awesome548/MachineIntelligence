import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

seq_len = 128
input_size = 3
train_size = 950
test_size = 200
test_minisize = 50
output_size = 2
hidden_size_1 = 50
hidden_size_2 = 70
epoch_num = 500
batch = 32
learning_rate = 0.001

"""
f = open("./Dataset/myData/person007.csv")
f_o = open("./Dataset/myData/person_007.csv", "w")

line = f.readline()
while line:
    f_o.writelines([line[:-7], "\n"])
    line = f.readline()

f.close()
f_o.close()
"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
path = "./Dataset/myData/"
mmscaler = MinMaxScaler(feature_range=(-1, 1), copy=True)

"""
Scale data
"""

#training set
# amaya_data = np.loadtxt(path + "amaya0701_out.csv", delimiter=",", skiprows=100)
# kikuzo_data = np.loadtxt(path + "kikuzo0701_out.csv", delimiter=",", skiprows=100)
# rinto_data = np.loadtxt(path + "rinto0701_out.csv", delimiter=",", skiprows=100)
person000 = np.loadtxt(path + "person_000.csv", delimiter=",", skiprows=50)
person001 = np.loadtxt(path + "person_001.csv", delimiter=",", skiprows=50)
person002 = np.loadtxt(path + "person_002.csv", delimiter=",", skiprows=50)
person003 = np.loadtxt(path + "person_003.csv", delimiter=",", skiprows=50)
person004 = np.loadtxt(path + "person_004.csv", delimiter=",", skiprows=50)
person005 = np.loadtxt(path + "person_005.csv", delimiter=",", skiprows=50)
person006 = np.loadtxt(path + "person_006.csv", delimiter=",", skiprows=50)
person007 = np.loadtxt(path + "person_007.csv", delimiter=",", skiprows=50)

d_000 = person000[:,1:4]
d_001 = person001[:,1:4]
d_002 = person002[:,1:4]
d_003 = person003[:,1:4]
d_004 = person004[:,1:4]
d_005 = person005[:,1:4]
d_006 = person006[:,1:4]
d_007 = person007[:,1:4]

data_000 = np.ones((int(len(d_000)/seq_len), seq_len, input_size), float)
data_001 = np.ones((int(len(d_001)/seq_len), seq_len, input_size), float)
data_002 = np.ones((int(len(d_002)/seq_len), seq_len, input_size), float)
data_003 = np.ones((int(len(d_003)/seq_len), seq_len, input_size), float)
data_004 = np.ones((int(len(d_004)/seq_len), seq_len, input_size), float)
data_005 = np.ones((int(len(d_005)/seq_len), seq_len, input_size), float)
data_006 = np.ones((int(len(d_006)/seq_len), seq_len, input_size), float)
data_007 = np.ones((int(len(d_007)/seq_len), seq_len, input_size), float)

for i in range(int(len(d_000)/seq_len)):
    data_000[i,:,:] = d_000[i*seq_len:(i+1)*seq_len,:]
for i in range(int(len(d_001)/seq_len)):
    data_001[i,:,:] = d_001[i*seq_len:(i+1)*seq_len,:]
for i in range(int(len(d_002)/seq_len)):
    data_002[i,:,:] = d_002[i*seq_len:(i+1)*seq_len,:]
for i in range(int(len(d_003)/seq_len)):
    data_003[i,:,:] = d_003[i*seq_len:(i+1)*seq_len,:]
for i in range(int(len(d_004)/seq_len)):
    data_004[i,:,:] = d_004[i*seq_len:(i+1)*seq_len,:]
for i in range(int(len(d_005)/seq_len)):
    data_005[i,:,:] = d_005[i*seq_len:(i+1)*seq_len,:]
for i in range(int(len(d_006)/seq_len)):
    data_006[i,:,:] = d_006[i*seq_len:(i+1)*seq_len,:]
for i in range(int(len(d_007)/seq_len)):
    data_007[i,:,:] = d_007[i*seq_len:(i+1)*seq_len,:]

    
print(data_000.shape)
print(data_001.shape)
print(data_002.shape)
print(data_003.shape)
print(data_004.shape)
print(data_005.shape)
print(data_006.shape)
print(data_007.shape)

train000 = data_000[:100]
train001 = data_001[:350]
train002 = data_002[:100]
train003 = data_003[:100]
train004 = data_004[:100]
train005 = data_005[:100]
train006 = data_006[:100]
train007 = data_007[:100]

trX = np.vstack([train001, train000])
trX = np.vstack([trX, train002])
trX = np.vstack([trX, train003])
trX = np.vstack([trX, train004])
trX = np.vstack([trX, train005])
trX = np.vstack([trX, train006])
print(trX.shape)

test001 = data_001[350:450]
test006 = data_006[:100]
test007 = data_007[:100]

teX = np.vstack([test001, test007])
print(teX.shape)

trY = np.zeros(train_size, int)
trY[350:950] = np.ones(600, int)
print(trY.shape)

teY = np.zeros(test_size, int)
teY[100:200] = np.ones(100,int)
print(teY.shape)

trainX = torch.Tensor(np.array(trX)).to(device)
trainY = torch.Tensor(np.array(trY)).to(device)

testX = torch.Tensor(np.array(teX)).to(device)
testY = torch.Tensor(np.array(teY)).to(device)

print('trainX.shape:{0}'.format(trainX.shape))
print('trainY.shape:{0}'.format(trainY.shape))
print('testX.shape:{0}'.format(testX.shape))
print('testY.shape:{0}'.format(testY.shape))

# trainYをone-hotにするためlongにsuru
trainY = trainY.long()
# one-hotにする 
trainY = F.one_hot(trainY, num_classes=output_size)
# 誤差を計算できるようにfloatに直す
trainY = trainY.float()
# trY one-hotにする
trY = trainY.cpu().data.numpy()

#testY one-hot ni sinai!
print('trainY.shape:{0}'.format(trainY.shape))
print('trainY: {0}'.format(trainY))
print('testY: {0}'.format(testY))

"""
DataLoader
"""
class DataSet:
    def __init__(self):
        self.X = trX.astype(np.float32) # 入力
        self.t = trY # 出力

    def __len__(self):
        return len(self.X) # データ数(10)を返す

    def __getitem__(self, index):
        # index番目の入出力ペアを返す
        return self.X[index], self.t[index]

# さっき作ったDataSetクラスのインスタンスを作成
dataset = DataSet()
# datasetをDataLoaderの引数とすることでミニバッチを作成．
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, \
                                         shuffle=True, drop_last=True)

"""
Model
"""
class MyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.cnn = CNN()
        self.lstm_1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size_1, \
                              num_layers=1, batch_first=True) 
        self.lstm_2 = nn.LSTM(input_size=self.hidden_size_1, hidden_size=self.hidden_size_2, \
                             num_layers=1, batch_first=True) 
        self.relu = nn.ReLU()
        #self.linear = nn.Linear(self.hidden_size_1, output_size)
        self.linear = nn.Linear(self.hidden_size_2, output_size)
        #self.softmax = nn.Softmax(-1)

    def forward(self, x):
        # self.cnn(x)
        h_0 = torch.zeros(1, x.size(0), self.hidden_size_1).to(device)
        c_0 = torch.zeros(1, x.size(0), self.hidden_size_1).to(device)
        h_1 = torch.zeros(1, x.size(0), self.hidden_size_2).to(device)
        c_1 = torch.zeros(1, x.size(0), self.hidden_size_2).to(device)
        out, (h_out, c_out) = self.lstm_1(x, (h_0, c_0))
        _, (h_out, _) = self.lstm_2(out, (h_1, c_1))
        #h_out = h_out.view(-1, self.hidden_size_1)
        h_out = h_out.view(-1, self.hidden_size_2)
        h_out = self.relu(h_out)
        y_hat = self.linear(h_out)
        #y_hat = self.softmax(h_out)
        return y_hat

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = input_size
        # cnn
        
        self.relu = nn.ReLU()
        

    def forward(self, x):
        
        # cnn
        h_out = self.relu(h_out)
        
        
        return y_hat

def train(model, optimizer, X, t):
  model.train()
  y_hat = model(X)
  # print(y_hat.shape)
  # loss = F.mse_loss(y_hat, trainY)
  loss = nn.CrossEntropyLoss()
  output = loss(y_hat, t)
  optimizer.zero_grad()
  output.backward()
  optimizer.step()
  return output.item()

loss = []


def main():
  model = MyLSTM()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  model = model.to(device)
  for epoch in range(epoch_num):
    for X, t in dataloader:
      _loss = train(model, optimizer, X.to(device), t.to(device))
      loss.append(_loss)
    if epoch % 20 == 0:
      print(f"Epoch = {epoch+1}, Loss = {_loss:.5f}")
  return model

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

# create default optimizer for doctests

param_tensor = torch.zeros([1], requires_grad=True)
default_optimizer = torch.optim.SGD([param_tensor], lr=learning_rate)

# create default trainer for doctests
# as handlers could be attached to the trainer,
# each test must define his own trainer using `.. testsetup:`

def get_default_trainer():

    def train_step(engine, batch):
        return batch

    return Engine(train_step)

# create default model for doctests

# default_model = nn.Sequential(OrderedDict([
#     ('base', nn.Linear(4, 2)),
#     ('fc', nn.Linear(2, 1))
# ]))

# manual_seed(666)

def predict(model):
  model.eval()
  train_predict = model(testX)
  print(train_predict.shape)
  
  data_predict = torch.argmax(train_predict, dim=1)
  data_predict = F.one_hot(data_predict, num_classes=output_size)
  


model = main()

predict(model)

