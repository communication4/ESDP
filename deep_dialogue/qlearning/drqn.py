
import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DRQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DRQN, self).__init__()
        self.state_space = input_size
        self.hidden_space = hidden_size
        self.output_size = output_size
        self.num_layers = 1
        self.lstm = nn.LSTM(self.state_space, self.hidden_space, self.num_layers)
        self.Linear = nn.Linear(self.hidden_space, self.output_size)


    def forward(self, x):  # 多了一层网络结构
        # print("xx---",x.shape)

        if len(x.shape) != 3:
            x = x.unsqueeze(0).to(device)
            self.batch_size = x.shape[1]
            # print("xx---", x.shape)
        else:
            self.batch_size = x.shape[1]


        # print("self.batch_size::....",self.batch_size)

        self.h = torch.zeros(self.num_layers, self.batch_size, self.hidden_space).to(device)
        self.c = torch.zeros(self.num_layers, self.batch_size, self.hidden_space).to(device)

        out, _ = self.lstm(x, (self.h, self.c))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out1 = self.Linear(out)  # 此处的-1说明我们只取RNN最后输出的那个hn
        # print('shape------',out1.shape)  # torch.Size([1, 1, 43])
        return out1.squeeze(0)



    def predict(self, x):
        y = self.forward(x)
        return torch.argmax(y, 1)