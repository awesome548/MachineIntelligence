import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torchinfo import summary


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
epoch_num = 75
batch = 16
learning_rate = 0.01

owner = True
theif = False
model_path = 'model.pth'
path = "./Dataset/myData/"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(data):
    data = data[np.newaxis,:,:]
    testX = torch.Tensor(data).to(device)
    print(testX.shape)

    model = MyLSTM()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    result = predict(model, testX)

    return result


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


def predict(model, test):
    model.eval()
    train_predict = model(test)
    train_predict = torch.argmax(train_predict, dim=1)
    #print(train_predict)
    if train_predict == 0:
        print("Owner")
        return 0
    else :
        print("theif")
        return 1
    

# def temp_test(bool):
#     person000 = np.loadtxt(path + "person_000.csv", delimiter=",", skiprows=50)
#     person005 = np.loadtxt(path + "person_005.csv", delimiter=",", skiprows=50)
#     d_000 = person000[:,1:7]
#     d_005 = person005[:,1:7]
#     data_000 = np.ones((int(len(d_000)/seq_len), seq_len, input_size), float)
#     data_005 = np.ones((int(len(d_005)/seq_len), seq_len, input_size), float)
#     for i in range(int(len(d_000)/seq_len)):
#         data_000[i,:,:] = d_000[i*seq_len:(i+1)*seq_len,:]
#     for i in range(int(len(d_005)/seq_len)):
#         data_005[i,:,:] = d_005[i*seq_len:(i+1)*seq_len,:]
#     test_true = torch.Tensor(np.array([data_000[400]])).to(device)
#     test_false = torch.Tensor(np.array([data_005[2]])).to(device)
#     if bool:
#         return test_true
#     else :
#         return test_false
"""
stop = len(loss)
step = int(len(loss) / epoch_num)
plt.plot(loss[0:stop:step], '.', label = "test_error")
plt.show()
"""
