import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable


class EmoNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, output_size):
        super(EmoNet, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_s2h = nn.Linear(self.state_size, self.hidden_size)
        self.linear_a2h = nn.Linear(self.action_size, self.hidden_size)
        self.linear_h2o = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, s, a):
        x_s = F.tanh(self.linear_s2h(s))
        x_a = F.tanh(self.linear_a2h(a))
        x = F.sigmoid(self.linear_h2o(x_s + x_a))
        return x


