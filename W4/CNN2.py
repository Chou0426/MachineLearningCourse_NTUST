import torch
from torch.utils.data import DataLoader
from torchvision import transforms , datasets
from torch import nn,optim
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

train_path = 'dogs-vs-cats/train'
test_path = 'dogs-vs-cats/test1'

train_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
test_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.5, ),(0.5, ))])

train_data = datasets.ImageFolder(train_path,transform=train_transform)
test_data = datasets.ImageFolder(test_path,transform=test_transform)

train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
test_loader = DataLoader(test_data,batch_size=64,shuffle=True)


print('label:',train_data.class_to_idx)
print('path & label:',train_data.imgs[0])
print('image')
print(train_data[0][0])
print(train_data[0][1])
print('size:',train_data[0][0].shape)


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64,128,3,1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128*26*26,512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512,2)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = x.view(-1,128*26*26)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x 

device = torch.device('cpu')
model = CNN()
loss_f = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr  = 1e-3)


def train():
    model.train()
    train_loss = 0

    for i , (data,label) in enumerate(train_loader):
        data , label = data.to(device) , label.to(device)
        pred = model(data)
        loss = loss_f(pred,label)


        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)


def test():
    model.eval()
    test_loss = 0

    for i,(data,label) in enumerate(test_loader):
        data , label = data.to(device),label.to(device)
        pred = model(data)
        loss = loss_f(pred,label)

        test_loss += loss.item()

    return test_loss / len(test_loader)


train_losses = []
test_losses = []

for i in range(0,2):
    train_loss = train()
    test_loss = test()
    train_losses.append(train_loss)
    test_losses.append(test_loss)


    print('epoch:{},train_loss:{:.6f},test_loss:{:6f}'.format(i,train_loss,test_loss))

torch.save(model.state_dict(),'CNN.pt')
df = pd.DataFrame((train_losses,test_losses))
df = df.T
df.to_csv('loss.csv',header=0,index=0)


model.load_state_dict(torch.load('CNN.pt'))
model.eval()
img = Image.open('dogs-vs-cats/test/test1/1.jpg').con
data = train_transform(img)
data = torch.unsqueeze(data,dim=0)
pred = model(data)
_,y = torch.max(pred,1)


df = pd.read_csv('loss.csv',header=None)
train_loss =np.array(df[0])
test_loss= np.array(df[1])

plt.figure(1)
plt.imshow(img)
plt.title('cat' if y.cpu().numpy()==0 else 'dog')

plt.figure(2)
plt.plot(train_loss)
plt.plot(test_loss)
plt.show()