from torch.autograd import Variable
import torch
import numpy as np
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torch import nn,optim
import matplotlib.pyplot as plt


# data_transform 有內建的，但自己寫會比較有彈性
def data_transform(x):
    x = np.array(x,dtype='float32')/255 # x = 0~1
    x = (x-0.5)/0.5   # x = -1~1
    x = x.reshape((-1,))
    x = torch.from_numpy(x)

    return x

class MLP(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim,500)  # (輸入層的size,輸出的神經元數量)
        self.linear2 = nn.Linear(500,250)        # (上一層的輸出,此層的神經元數量)
        self.linear3 = nn.Linear(250,125)
        self.linear4 = nn.Linear(125,output_dim)  # 有幾個輸出類別,輸出層就會有多少神經元
        self.fc = nn.Softmax()  # Fully Connected layer

    def forward(self,x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.fc(x)

        return x 

train_data = mnist.MNIST('./mnist',train=True,transform = data_transform,download=False)
test_data = mnist.MNIST('./mnist',train=False,transform = data_transform,download=False)



train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
test_loader = DataLoader(test_data,batch_size=64,shuffle=True)


model = MLP(784,10)
print(model)
loss_f = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr = 1e-5) # lr會自動調整
epoch = 50

def train():
    train_loss = 0
    train_acc = 0
    model.train()
    for im,label in train_loader:
        im = Variable(im)
        label = Variable(label)

        pred = model(im)
        loss = loss_f(pred,label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss
        _ , y = pred.max(1)
        correct = (y==label).sum().item()
        acc = correct / im.shape[0]
        train_acc += acc
    print(pred)    
    print(label)  # tensor([[0., 0., 1.]])
    print(y)  # tensor([[0])
    print(correct)
    return train_loss,train_acc

def test():
    test_loss = 0
    test_acc = 0
    model.eval()  # 將最好的model拿來用

    for im,label in test_loader:
        im = Variable(im)
        label = Variable(label)

        pred = model(im)
        loss = loss_f(pred,label)
        test_loss += loss

        _ , y = pred.max(1)
        correct = (y==label).sum().item()
        
        acc = correct / im.shape[0]
        test_acc += acc

    return test_loss , test_acc

train_losses = []
test_losses = []
train_acces = []
test_acces = []

for i in range(0,3):
    train_loss , train_acc = train()
    test_loss, test_acc = train()

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_acces.append(train_acc)
    test_acces.append(test_acc)

    print('epoch:{}, train_loss:{:.6f},train_acc:{:.6f},test_loss:{:.6f},test_acc:{:.6f}'.format(i, train_loss/len(train_loader), train_acc/len(train_loader), test_loss/len(test_loader), test_acc/len(test_loader)))


fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(train_losses,'r')
ax.plot(test_losses,'b')
ax.set_ylabel('loss')
ax.legend(['train_loss','test_loss'],loc = 'upper right')

ax1 = ax.twinx()
ax1.plot(train_acces,'g')
ax1.plot(test_acc,'y')
ax1.set_ylabel('acc')
ax1.legend(['train_acc','test_acc'],loc = 'lower right')

plt.show()