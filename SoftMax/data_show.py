import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as Data


def get_dataiter(train=True,batch=256,shuffle=True):
    train_minist,test_minist=download_dataset()
    if train is True:
        data_iter=Data.DataLoader(train_minist,batch,shuffle)
        return data_iter
    else:
        data_iter=Data.DataLoader(test_minist,batch,shuffle)
        return data_iter


def show_data(features, labels):
    _, figs = plt.subplots(1, len(features),figsize=(12,12))
    for fig,feature,label in zip(figs,features,labels):
        fig.imshow(feature.view((28,28)).numpy())
        fig.set_title(label)

    plt.show()


def download_dataset():
    """
    Download fashion minist dataset from github and store it in ~/Datasets/FashionMNIST. If it exist, return train
    data and test data.
    Data format:
    length is 60000(train)/10000(test) and every element is tuple (tensor,int) every tensor represents picture
    which is 1*28*28 (channel height width) or (C*H*W) which is transformed by (H*W*C)
    :return: train_dataset,test_dataset
    """
    minist_train = torchvision.datasets.FashionMNIST(root="~/Datasets/FashionMNIST", train=True, download=True,
                                                     transform=transforms.ToTensor())
    minist_test = torchvision.datasets.FashionMNIST(root="~/Datasets/FashionMNIST", train=False, download=True,
                                                    transform=transforms.ToTensor())
    return minist_train, minist_test


def labels2language(labels):
    language = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [language[i] for i in labels]


def main():
    minist_train, minist_test = download_dataset()
    ind_list = range(10)
    features=[]
    labels=[]
    for ind in ind_list:
        features.append(minist_train[ind][0])
        labels.append(minist_train[ind][1])
    labels_lan = labels2language(labels)
    show_data(features, labels_lan)

if __name__ == "__main__":
    main()