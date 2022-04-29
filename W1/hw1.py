import torch 
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn , optim

epoch = 1000
df = pd.read_csv('pokemon.csv')
x = np.array(df['cp'])
y = np.array(df['cp_new'])

print(x,y)

x = np.reshape(x, (x.size,1))
y = np.reshape(y, (y.size,1))
x = x.astype(np.float32)
y = y.astype(np.float32)

x = torch.tensor(x)
y = torch.tensor(y)

print(x,y)

class Model(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim,output_dim)
    
    def forward(self,x):
        x = self.linear(x)
        return x 

model = Model(1,1)
opt = optim.Adam(model.parameters(),lr = 0.01)
losses = []

for i in range(epoch):
    y_hat = model(x)
    loss_f = nn.MSELoss()
    loss = loss_f(y,y_hat)
    loss.backward()

    opt.step()
    opt.zero_grad()
    losses.append(loss)

    print("epoch:{} , loss:{} ".format(i,loss))

z = model(x).detach().numpy()
print(z)
plt.figure(1)
plt.plot(losses,'r')

plt.figure(2)
plt.plot(x,y, 'o' , 'r')
plt.plot(x,z)
plt.show()





