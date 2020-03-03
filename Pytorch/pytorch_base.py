# Zhu Jianing  2019.12.12
# Pytorch practice: basic tensor operation

import torch
import torch.nn as nn 
import numpy as np

x = torch.empty(5,3)
x = torch.zeros(5,3, dtype=torch.long)
x = x.new_ones(5,3, dtype=torch.double)
x = torch.randn_like(x, dtype=torch.double)

y = torch.rand(5,3, dtype=torch.double)
z = torch.add(x,y)
y.add_(x)   #result is in y

x = torch.randn(4,4)
y = x.view(16)  # reshape or resize operation

x = torch.randn(1)
x.item()

z = torch.ones(5)
zz = z.numpy()  # change tensor to numpy
zz = torch.from_numpy(zz)   # change numpy to tensor

# use GPU to calculate
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x,device=device)
    x = x.to(device)
    z = x + y

# use numpy to build easy NN

# Relu & L2 Loss
# h = wx 
# a = max(0, h)
# y = wa 

N , Din, H, Dout = 64, 1000, 100, 10
x = np.random.randn(N, Din)
y = np.random.randn(N, Dout)
w1 = np.random.randn(Din, H)
w2 = np.random.randn(H, Dout)

l_r = 1e-6
for e in range(500):
    #forward
    h = x.dot(w1)
    h_re = np.maximum(h, 0)
    y_pre = h_re.dot(w2)

    #loss: l2
    l2 = np.square(y_pre - y).sum()
 #   print(e, l2)
    
    #backward: Calculate the gradient
    grad_y_pre = 2.0 * (y_pre - y)
    grad_w2 = h_re.T.dot(grad_y_pre)
    grad_h_re = grad_y_pre.dot(w2.T)
    grad_h = grad_h_re.copy()
    grad_h[h<0] = 0
    grad_w1 = x.T.dot(grad_h)

    #update w
    w1 -= l_r * grad_w1
    w2 -= l_r * grad_w2


# use tensor to build easy NN
N , Din, H, Dout = 64, 1000, 100, 10
x = torch.randn(N, Din)
y = torch.randn(N, Dout)
w1 = torch.randn(Din, H, requires_grad=True)
w2 = torch.randn(H, Dout, requires_grad=True)

l_r = 1e-6
for e in range(500):
    #forward
    h = x.mm(w1)
    h_re = h.clamp(min=0)
    y_pre = h_re.mm(w2)

    #loss: l2
    l2 = (y_pre - y).pow(2).sum()
  #  print(e, l2.item())
    

    #backward: Calculate the gradient
    l2.backward()

    #update w
    with torch.no_grad():
        w1 -= l_r * w1.grad
        w2 -= l_r * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()

# use torch.nn build easy NN

N , Din, H, Dout = 64, 1000, 100, 10
x = torch.randn(N, Din)
y = torch.randn(N, Dout)

class ZNN (torch.nn.Module):
    def __init__(self, Din, H, Dout):
        super(ZNN, self).__init__()
        self.linear1 = nn.Linear(Din, H, bias=False)
        self.linear2 = nn.Linear(H, Dout, bias=False)
    
    def forward(self, x):
        y_pre = self.linear2(self.linear1(x).relu())
        return y_pre

model = ZNN(Din, H, Dout)
#model = model.cuda()
l2 = nn.MSELoss(reduction='sum')
l_r = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=l_r)
for e in range(500):
    #forward
    y_pre = model(x) 

    #loss: l2
    l22 = l2(y_pre, y)
    print(e, l22.item())
    
    optimizer.zero_grad()
    #backward: Calculate the gradient
    l22.backward()

    #update w
    optimizer.step()

