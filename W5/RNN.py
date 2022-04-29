import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torch import nn,optim

x = np.arange(1,721,1)
y = np.sin(x * np.pi / 180)  + np.random.randn(720)*0.05

X = []
Y = []

for i in range(0,710):
    list1 = []
    for j in range(i,i+10):
        list1.append(y[j])
    X.append(list1)
    Y.append(y[j+1])


X = np.array(X)
Y = np.array(Y)

train_X = X[:360]
train_Y = Y[:360]
test_X = X[360:]
test_Y = Y[360:]


class timeseries(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

    def __len__(self):
        return self.len

dataset  = timeseries(train_X,train_Y)
train_loader = DataLoader(dataset, shuffle=True,  batch_size=64)

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(input_size = 1,hidden_size = 5 ,num_layers = 1 ,batch_first = True)
        self.linear = nn.Linear(5,1)

    def forward(self,x):
        r_out , h_state = self.rnn(x)
        out = r_out[:,-1,:]
        out = self.linear(out)
        
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RNN().to(device)
loss_f = nn.MSELoss()
opt = optim.Adam(model.parameters(),lr=1e-3)


epoch = 100
losses = []

for i in range(epoch):
    for index,data in enumerate(train_loader):
        pred = model(data[:][0].view(-1,10,1)).reshape(-1)
        loss = loss_f(pred,data[:][1])
        loss.backward()
        opt.step()
        losses.append(loss)

    if i%10 == 0:
        print("epoch:{},loss:{:.6f}".format(i,loss.item()))

test_data = timeseries(test_X,test_Y)
pred = model(test_data[:][0].view(-1,10,1)).view(-1)

plt.figure(1)
plt.title("Loss")
plt.plot(losses)
plt.figure(2)
plt.plot(pred.detach().numpy(),label='pred')
plt.plot(test_data[:][1].view(-1),label='org')
plt.legend()
plt.show()