import torch.nn as nn
class MyLinearNet(nn.Module):
    def __init__(self,input_num,output_num):
        super(MyLinearNet,self).__init__()
        self.first_layer=nn.Linear(input_num,output_num)

    def forward(self, X):
        y=self.first_layer(X)
        return y