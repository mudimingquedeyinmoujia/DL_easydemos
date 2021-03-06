import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import model


def softmax(x):
    x_exp = x.exp()
    partition = x_exp.sum(dim=1, keepdim=True)
    return x_exp / partition


def cal_acc(test_iter, net):
    acc_sum = 0.0
    n = 0
    for features, labels in test_iter:
        output_raw = net(features)
        y_hat = softmax(output_raw)
        acc = (y_hat.argmax(dim=1) == labels).float().sum().item()
        acc_sum += acc
        n += labels.shape[0]
    return acc_sum / n


def train(train_iter, test_iter, input_n, output_n, epochs, loss, lra):
    net = model.MySoftModel(input_n, output_n)
    init.normal_(net.linear.weight, mean=0, std=1)
    init.constant_(net.linear.bias, 0)
    optimizer = optim.SGD(net.parameters(), lr=lra)
    acc_rate = cal_acc(test_iter, net)
    print("not train acc:{}".format(acc_rate))
    for epoch in range(1, epochs + 1):
        print("epoch {} start".format(epoch))
        for features, labels in train_iter:
            outputs = net(features)
            l = loss(outputs, labels)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        acc_rate = cal_acc(test_iter, net)
        print("epoch {}, loss:{}, acc_rate:{}".format(epoch, l.item(), acc_rate))
