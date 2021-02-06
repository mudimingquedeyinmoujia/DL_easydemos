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
    with torch.no_grad():
        for features, labels in test_iter:
            output_raw = net(features)
            y_hat = softmax(output_raw)
            acc = (y_hat.argmax(dim=1) == labels).float().sum().item()
            acc_sum += acc
            n += labels.shape[0]
    return acc_sum / n


def train(train_iter, test_iter, input_n, output_n, epochs, loss, lra):
    # net = model.MySoftModel(input_n, output_n)
    net = model.LeNet()
    for params in net.parameters():
        init.normal_(params, mean=0, std=1)

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

        acc_rate_test = cal_acc(test_iter, net)
        acc_rate_train = cal_acc(train_iter, net)
        print("epoch {}, loss:{}, acc_rate_train:{}, acc_rate_test:{}".format(epoch, l.item(), acc_rate_train,
                                                                              acc_rate_test))
