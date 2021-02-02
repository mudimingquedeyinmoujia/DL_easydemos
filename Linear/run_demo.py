import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import model
import torch.nn.init as init

# prepare data
# y=w0*x0+w1*x1+w2*x2+b
sample_num = 1000
true_w = [2.4, 3.8, 12.3]
true_b = 4.1
features = torch.tensor(np.random.normal(0, 5, (sample_num, 3)), dtype=torch.float32)
labels = features.mm(torch.tensor(true_w).view(-1, 1)) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, (labels.size())))

dataset = Data.TensorDataset(features, labels)
dataiter = Data.DataLoader(dataset, batch_size=10, shuffle=True)

# 准备loss
loss = nn.MSELoss()

# 初始化model
net = model.MyLinearNet(input_num=3, output_num=1)
init.normal_(net.first_layer.weight, mean=0, std=1)
# 准备优化
optimizer = optim.SGD(net.parameters(), lr=0.01)  # 注意不是weight而是parameters()

# 开始训练
epochs = 5
for epoch in range(1, epochs + 1):
    print("epcch {} start".format(epoch))
    for ite in dataiter:
        features_batch, labels_batch = ite
        output = net(features_batch)
        optimizer.zero_grad()
        l = loss(output, labels_batch)
        l.backward()
        optimizer.step()
    print("epoch {} is ok and the loss is {}".format(epoch, l.item()))

dense = net.first_layer
print("the ground truth is w {} {} {} b {}".format(true_w[0], true_w[1], true_w[2], true_b))
print("the final result is w {} b {}".format(dense.weight, dense.bias))
