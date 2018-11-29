#autograd:自动微分
#PyTorch中所有神经网络的核心是autograd包。让我们先简单地看一下这个，然后我们将去训练我们的第一个神经网络。
#autograd包为张量上的所有操作提供自动微分。它是一个按运行定义的框架，这意味着您的支持是由代码的运行方式定义的，并且每个迭代都可能不同。
#让我们用一些简单的例子来看看这个问题。

import torch

#requires_grad = True 追踪计算
x = torch.ones(2,2,requires_grad=True)
print(x)

y = x + 2
print(y)

#因为y是被创造的操作的结果，所以它有grad_fn
print(y.grad_fn)

z = y*y*3
out = z.mean()

print(z,out)
#z,out都会有grad_fn

#默认的requires_grad是False
a = torch.randn(2,2)
a = ((a*3)/(a-1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a*a).sum()
print(b.grad_fn)

#反向传播 
out.backward()

#代表d(out)/dx
print(x.grad)

#norm()函数是求范数的公式
x = torch.randn(3,requires_grad=True)
y = x*2
while y.data.norm() < 1000:
    y = y*2
print(y)

gradients = torch.tensor([0.1 1.0 0.0001],dtype = torch.float)
y.backward(gradients)

print(x.grad)


print(x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad():
    print((x**2).requires_grad)


