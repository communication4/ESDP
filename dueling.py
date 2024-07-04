import torch
import torch.nn as nn

'''优势更新，此处并没有设置优化回放，他们使用的经验池还是一样的'''


class DQN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # self.linear_i2h = nn.Linear(self.input_size, self.hidden_size)
        # self.linear_h2o = nn.Linear(self.hidden_size, self.output_size)

        self.feature = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size))

        self.value = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1))

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        out = value + advantage - advantage.mean()
        # print('shape------', out.shape)  # shape------ torch.Size([1, 1])
        return out  # 优势更新

    def predict(self, x):
        y = self.forward(x)
        # print(y.squeeze())
        # print("动作的长度：",len(y.squeeze()))
        return torch.argmax(y, 1)
        # return y.max(1)[1].data[0]
