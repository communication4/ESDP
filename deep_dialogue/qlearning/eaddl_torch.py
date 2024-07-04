import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable


class EADDL(nn.Module):
    def __init__(self, input_size_a, input_size_b, hidden_size, output_size):
        super(EADDL, self).__init__()

        self.input_size_a = input_size_a
        self.input_size_b = input_size_b
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_ia2vha = nn.Linear(self.input_size_a, self.hidden_size)
        self.linear_vha2va = nn.Linear(self.hidden_size, 1)

        self.linear_ib2vhb = nn.Linear(self.input_size_b, self.hidden_size)
        self.linear_vhb2vb = nn.Linear(self.hidden_size, 1)

        self.linear_i2ah = nn.Linear(self.input_size_a+self.input_size_b, self.hidden_size)
        self.linear_ah2a = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x_a, x_b):

        va = self.linear_vha2va(F.tanh(self.linear_ia2vha(x_a)))
        vb = self.linear_vhb2vb(F.tanh(self.linear_ib2vhb(x_b)))
        a = self.linear_ah2a(F.tanh(self.linear_i2ah(torch.cat([x_a, x_b], dim=-1))))

        return va.expand(a.size()) + vb.expand(a.size()) + a - a.mean(-1).unsqueeze(1).expand(a.size())


    def predict(self, x_a, x_b):
        y = self.forward(x_a, x_b)
        return torch.argmax(y, 1)


