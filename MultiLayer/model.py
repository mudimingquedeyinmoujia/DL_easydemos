import torch.nn as nn


class MySoftModel(nn.Module):
    def __init__(self, input_n, output_n):
        super(MySoftModel, self).__init__()
        self.linear = nn.Linear(input_n, output_n)

    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
    def forward(self, x):
        return x.view(x.shape[0],-1)