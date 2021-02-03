import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as Data
import torch.optim
import data_show
import model
import train

# prepare data
batch_size = 256
feature_input = 28 * 28
label_output = 10
data_iter = data_show.get_dataiter(train=True, batch=batch_size, shuffle=True)

# init model
net = model.MySoftModel(feature_input, label_output)
init.normal_(net.linear.weight, mean=0, std=1)
init.constant_(net.linear.bias, 0)

# loss
loss = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# 1. normal train
# epochs = 10
# for epoch in range(1, epochs + 1):
#     print("epoch {} is start".format(epoch))
#     for X, y in data_iter:
#         output = net(X)
#         l = loss(output, y)
#         optimizer.zero_grad()
#         l.backward()
#         optimizer.step()
#     print("epoch {} is done and loss is {}".format(epoch, l.item()))

# 2. use train.py to train
test_iter=data_show.get_dataiter(False,256,True)
train.train(data_iter,test_iter,feature_input,label_output,50,loss,0.05)