import torch

scalar=torch.tensor(7)
vector=torch.tensor([1,2,3])
matrix=torch.tensor([[1,2],[3,4]])

print("标量",scalar.item())
print("向量的形状：",vector.shape)
print("矩阵的形状：",matrix.shape)
A=torch.tensor([[1.,2.],[3.,4.]])
B=torch.tensor([[5.,6.],[7.,8.]])

C=torch.matmul(A,B)
print("矩阵A乘B的结果：\n",C)

input_tensor=torch.randn(3,2,4)

weight=torch.randn(4,5)

output=torch.matmul(input_tensor,weight)
print("变换后的形状：",output.shape)

x=torch.tensor(2.0)
w=torch.tensor(3.0,requires_grad=True)
y=w*x

y.backward()
print("w的梯度是：",w.grad)