"""
Liburary Set
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torchinfo import summary

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *

"""
SCALE DATA
"""
train_person = 0 # 0 = kikuzo, 1 = amaya, 2 = rinto
subject_num = 8
test_num = 2
seq_len = 128
input_size = 6
train_size = 350
test_size = 100
test_minisize = 50
output_size = 2

#MI detail
hidden_size_1 = 50
hidden_size_2 = 70
epoch_num = 200
batch = 16
learning_rate = 0.01

#OTHERS
path = "./Dataset/myData/"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# mmscaler = MinMaxScaler(feature_range=(-1, 1), copy=True)


"""
TRAIN-TEST SHEMA
"""
print("Train - Test labels : ")
temp = np.arange(subject_num)
temp = np.delete(temp,train_person)
np.random.shuffle(temp)
train_subject = temp[test_num:]
test_subject = temp[0:test_num]
train_subject.sort()
test_subject.sort()
print(train_subject, test_subject)

"""
DATA LOADING
"""
for i in range(subject_num):
    f = open(path + "person_00" + str(i) + ".csv")
    f_o = open(path + "person_00" + str(i) + "_out.csv", "w")

    line = f.readline()
    while line:
        f_o.writelines([line[:-7], "\n"])
        line = f.readline()

    f.close()
    f_o.close()

print("Dataset Size")
#scale data set
data = []
for i in range(subject_num):
    load = np.loadtxt(path + "person_00" + str(i) + "_out.csv", delimiter=",", skiprows=100)
    load = load[:,1:7]

    preData = np.zeros((int(len(load) / seq_len), seq_len, input_size), float)
    for j in range(int(len(load) / seq_len)):
        preData[j,:,:] = load[j * seq_len : (j + 1) * seq_len, :]
    
    data.append(preData)
    print("data", i, preData.shape)

#trainnig person : other = 1 : 1
trX = data[train_person][:train_size]
for i in train_subject:
    trX = np.vstack([trX, data[i][:train_size//len(train_subject)]])

trY = np.hstack([np.zeros(train_size, int), np.ones(train_size, int)])

teX = data[train_person][train_size:train_size + test_size]
for i in test_subject:
    teX = np.vstack([teX, data[i][:test_size // len(test_subject)]])

teY = np.hstack([np.zeros(test_size, int), np.ones(test_size, int)])

#print(trX.shape)
#print(teX.shape)
#print(trY.shape)
#print(teY.shape)

trainX = torch.Tensor(np.array(trX)).to(device)
trainY = torch.Tensor(np.array(trY)).to(device)

testX = torch.Tensor(np.array(teX)).to(device)
testY = torch.Tensor(np.array(teY)).to(device)

trainY = trainY.long()
trainY = F.one_hot(trainY, num_classes=output_size)
trainY = trainY.float()
trY = trainY.cpu().data.numpy()

#testY one-hot ni sinai!
print('trainY.shape:{0}'.format(trainY.shape))

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

class MyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.lstm_1 = nn.LSTM(input_size=6, hidden_size=self.hidden_size_1, num_layers=1, batch_first=True) 
        self.lstm_2 = nn.LSTM(input_size=self.hidden_size_1, hidden_size=self.hidden_size_2, \
                              num_layers=1, batch_first=True) 
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.hidden_size_2, output_size)
        #self.softmax = nn.Softmax(-1)

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
        #y_hat = self.softmax(h_out)
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

  #data_predict = train_predict.cpu().data.numpy()
  #testY_plot = testY.cpu().data.numpy()

  
  data_predict = torch.argmax(train_predict, dim=1)
  data_predict = F.one_hot(data_predict, num_classes=output_size)
  
  metric = ConfusionMatrix(num_classes=output_size)
  metric.attach(default_evaluator, 'cm')
  y_true = testY.int()
  y_pred = data_predict

  state = default_evaluator.run([[y_pred, y_true]])
  print(state.metrics['cm'])

model = main()
predict(model)
stop = len(loss)
step = int(len(loss) / epoch_num)
plt.plot(loss[0:stop:step], '.', label = "test_error")
plt.show()