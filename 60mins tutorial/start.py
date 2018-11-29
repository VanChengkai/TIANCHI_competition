from __future__ import print_function
import torch

#新建一个5x3的矩阵 未初始化
x = torch.empty(5,3)
print(x)

#构造一个随机初始化的矩阵
x = torch.rand(5,3)
print(x)

#构造一个零矩阵且类型为long
x = torch.zeros(5,3,dtype = torch.long)
print(x)

#直接从数据中构造一个张量
x = torch.Tensor([5.5,3])

#或者从一个存在的张量新建一个张量
x = x.new_ones(5,3,dtype=torch.double)
print(x)

x = torch.randn_like(x,dtype=torch.float)
print(x)


#获取x的size()
print(x.size())

#一些基本的操作
y = torch.rand(5,3)
print(x+y)

print(torch.add(x,y))

result = torch.empty(5,3)
torch.add(x,y,out=result)
print(result)

#如果要进行原地操作，在加法后把值赋给y
y.add_(x)
print(y)

#索引技巧 参考numpy
print(x[:,1])

#resize张量的尺寸
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8) #-1代表的维度自动填充
print(x.size(),y.size(),z.size())

#如果你有一个元素的张量，用.item()方法获取其Python的值
x = torch.randn(1)
print(x)
print(x,item())

#与numpy之间的转换
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

#因为共享内存
a.add_(1)
print(a)
print(b)

#将numpy的数组转换为torch的Tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)

#除了char张量之外，CPU上的所有张量都支持转换为NumPy

#使用CUDA 
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x,device = device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu",torch.double))





