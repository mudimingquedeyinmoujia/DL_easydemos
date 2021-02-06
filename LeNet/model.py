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
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # 28*28->24*24
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # 24*24->12*12
            nn.Conv2d(6, 16, 5),  # 12*12->8*8
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)  # 8*8->4*4
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120), # 这里不记录batch size的,也就是说接收的向量dim0不限!
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
