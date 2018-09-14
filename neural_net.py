#basic neural network with one hidden layer for approximating the cosine function

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

#data preparation
x=torch.FloatTensor([[i] for i in np.arange(-math.pi,math.pi+0.01,0.01)])
cosine=lambda x: torch.cos(x)
y=cosine(x)

fig=plt.figure()
plt.plot(x.numpy(),y.numpy())
fig.suptitle('Data')
plt.xlabel('x')
plt.ylabel('y=cos(x)')
plt.show()

#model training
y1=model(x)
for i in range(200):
    y_hat=model(x)
    loss=criterion(y_hat,y)
    print("epoch:",i,"; loss:",loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(x.detach().numpy(),y.detach().numpy(),'b',label='y')
plt.plot(x.detach().numpy(),y1.detach().numpy(),'r',label='y_before')
plt.plot(x.detach().numpy(),y_hat.detach().numpy(),'g',label='y_hat_after')
fig.suptitle('y vs y_hat')
plt.xlabel('x')
plt.ylabel('y and y_hat')
plt.legend(loc='upper right')
plt.show()
