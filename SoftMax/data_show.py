import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

minist_train=torchvision.datasets.FashionMNIST(root="~/Datasets/FashionMNIST",train=True,download=True,transform=transforms.ToTensor())
minist_test=torchvision.datasets.FashionMNIST(root="~/Datasets/FashionMNIST",train=False,download=True,transform=transforms.ToTensor())

print(type(minist_train))
print(len(minist_train),len(minist_test))