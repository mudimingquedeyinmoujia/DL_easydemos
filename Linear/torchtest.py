import torch
import numpy as np

# a=torch.tensor(np.arange(15).reshape(3,5))
# b=torch.tensor(np.arange(15).reshape(5,3))
# c=a.mm(b)
# print(a.t())
# print(b)
# print(c)

d=torch.tensor(np.random.normal(0,0.01,(3,5)))
print(d)