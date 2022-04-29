import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as data
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
epoch = 1

train_data = CIFAR10('./cifar10',train=True,transform = torchvision.transforms.ToTensor(),download=True)
test_data = CIFAR10('./cifar10',train=False,transform = torchvision.transforms.ToTensor(),download=True)


train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
test_loader = DataLoader(test_data,batch_size=64,shuffle=True)


class CNN(nn.Module):
    def __init__(self,input,output):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(input,16,3,1)  #(32-3)/1+1 = 30
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)  # 30/2 = 15
        self.conv2 = nn.Conv2d(16,32,3,1)  # (15-3)/1+1 = 13
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)  # 13/2 = 6.5 取 6
        self.fc1 = nn.Linear(32*6*6,100)
        self.fc2 = nn.Linear(100,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(-1,32*6*6)  # reshape
        x = self.fc1(x)
        x = self.fc2(x)

        return x 

device = torch.device('cpu')  # 有gpu可以用'cuda'
model = CNN(3,10)  # 輸入是3個channel
model = model.to(device)

loss_f = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(),lr=1e-3)

def train():
    model.train()
    train_loss = 0
    train_acc = 0
    for indx,(data,target) in enumerate(train_loader):   ## 一次讀一個batchsize,計算速度會變快,前面要加index    
        data,target = data.to(device),target.to(device)
        pred = model(data)

        loss = loss_f(pred,target)
        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss += loss
        _ , y = pred.max(1)
        corr = (y==target).sum().item()
        acc = corr/ data.shape[0]
        train_acc += acc

    return train_loss / len(train_loader) , train_acc / len(train_loader) 


def test():
    test_loss = 0
    test_acc = 0
    model.eval()

    for index,(data,target) in enumerate(test_loader):
        data,target = data.to(device),target.to(device)
        pred = model(data)
        loss = loss_f(pred,target)
        

        test_loss += loss
        _ , y = pred.max(1)
        corr = (y==target).sum().item()
        acc = corr / data.shape[0]
        test_acc += acc

    return test_loss / len(test_loader) , test_acc / len(test_loader)

train_losses = []
train_acces = []
test_losses = []
test_acces = []

for i in range(epoch+1):
    train_loss , train_acc = train()
    test_loss , test_acc = test()

    train_losses.append(train_loss)
    train_acces.append(train_acc)
    test_losses.append(test_loss)
    test_acces.append(test_acc)

    print('epoch:{} , train_loss:{:.6f},train_acc:{:.6f},test_loss:{:.6f},test_acc:{:.6f}'.format(i,train_loss,train_acc,test_loss,test_acc))




fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(train_losses,'r')
ax.plot(test_losses,'b')
ax.set_ylabel('loss')
ax.legend(['train_loss','test_loss'],loc = 'upper right')

ax1 = ax.twinx()
ax1.plot(train_acces,'g')
ax1.plot(test_acces,'y')
ax1.set_ylabel('acc')
ax1.legend(['train_acc','test_acc'],loc = 'lower right')


test_x = torch.tensor(test_data.data, dtype = torch.float32)
test_x /= 255
test_x = test_x.permute([0, 3, 1, 2])
#test_x = torch.unsqueeze(torch.tensor(test_data.data) , dim = 1).type(torch.FloatTensor)[:10]/255
test_output = model(test_x)
pred = torch.max(test_output,1)[1].data.numpy().squeeze()

fig2 = plt.figure(2,figsize=(10,4))
for i in range(0,10):
    ax2 = fig2.add_subplot(2,10/2,i+1)
    ax2.imshow(np.squeeze(test_x[i].permute([1, 2, 0])))
    ax2.set_title('pred:'+str(pred[i].item()))

plt.show()