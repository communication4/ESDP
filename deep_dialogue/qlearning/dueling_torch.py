import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable


class Dueling(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Dueling, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_i2ah = nn.Linear(self.input_size, self.hidden_size)
        self.linear_ah2a = nn.Linear(self.hidden_size, self.output_size)

        self.linear_i2vh = nn.Linear(self.input_size, self.hidden_size)
        self.linear_vh2v = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        v = self.linear_vh2v(F.tanh(self.linear_i2vh(x)))
        a = self.linear_ah2a(F.tanh(self.linear_i2ah(x)))

        return v.expand(a.size()) + a - a.mean(-1).unsqueeze(1).expand(a.size())

    def predict(self, x):
        y = self.forward(x)
        return torch.argmax(y, 1)

