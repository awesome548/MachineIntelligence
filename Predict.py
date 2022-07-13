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
epoch_num = 75
batch = 16
learning_rate = 0.01

#OTHERS
owner = True
theif = False
model_path = 'model.pth'
path = "./Dataset/myData/"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"""
MI architecture
"""
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
        return "Owner"
    else :
        print("theif")
        return "Theif"


def main(data):
    data = data[np.newaxis,:,:]
    testX = torch.Tensor(data).to(device)
    print(testX.shape)

    model = MyLSTM()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    result = predict(model, testX)

    return result


"""
stop = len(loss)
step = int(len(loss) / epoch_num)
plt.plot(loss[0:stop:step], '.', label = "test_error")
plt.show()
"""
