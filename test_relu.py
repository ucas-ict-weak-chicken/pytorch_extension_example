
import torch
from relu import MyReLU

device = torch.device('cuda')
dtype = torch.float
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device = device, dtype = dtype)
y = torch.randn(N, D_out, device = device, dtype = dtype)

w1 = torch.randn(D_in, H, device = device, dtype =dtype, requires_grad = True)
w2 = torch.randn(H, D_out, device = device, dtype =dtype, requires_grad = True)

lr = 1e-6
relu = MyReLU()
for i in range(500):
    h = relu(x.mm(w1))
    y_pred = h.mm(w2)

    loss = (y_pred - y).pow(2).sum()

    print('i:', i, 'loss:', loss.item())

    loss.backward()

    with torch.no_grad():
        w1 -= lr* w1.grad
        w2 -= lr* w2.grad

        w1.grad.zero_()
        w2.grad.zero_()

print('done!')
