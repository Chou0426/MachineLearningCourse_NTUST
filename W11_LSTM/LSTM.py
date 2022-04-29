import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torch import nn,optim
import matplotlib.pyplot as plt

# read file
df_data = pd.read_csv("./W11_LSTM/temperature.csv",header=0)

scaler = MinMaxScaler(feature_range=(-1,1))

'''Creat our dataset'''
class TemperatureDataset(Dataset):
    def __init__(self,data):
        self.df_temperature = data
        self.org_data = self.df_temperature.mean_temp.to_numpy()
        self.normalized_data = np.copy(self.org_data)
        self.normalized_data = self.normalized_data.reshape(-1,1)
        self.normalized_data = scaler.fit_transform(self.normalized_data)
        self.normalized_data = self.normalized_data.reshape(-1)
        self.sample_len = 12

    def __len__(self):
        if len(self.org_data) > self.sample_len:
            return len(self.org_data) - self.sample_len

        else:
            return 0

    def __getitem__(self, index):
        target = self.normalized_data[self.sample_len + index]
        target = np.array(target).astype(np.float64)


        i = self.normalized_data[index:(index + self.sample_len)]
        i = i.reshape(-1,1)
        i = torch.from_numpy(i)
        target = torch.from_numpy(target)
        

        return i , target

'''Dataset split'''
dataset = TemperatureDataset(df_data)
train_len = int(0.7 * len(dataset))
test_len = len(dataset) - train_len

train_data , test_data = random_split(dataset , [train_len,test_len])
train_loader = DataLoader(train_data,shuffle=False,batch_size=32)
test_loader = DataLoader(test_data,shuffle=False,batch_size=32)

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM,self).__init__()
        self.lstm = nn.LSTM(input_size = 1, hidden_size = 500, num_layers = 3, dropout = 0.1, batch_first = True)
        self.linear = nn.Linear(500,1)

    def forward(self,x):
        h_0 = torch.zeros([3, x.shape[0], 500], dtype=torch.double,device=x.device)
        c_0 = torch.zeros([3, x.shape[0], 500], dtype=torch.double,device=x.device)

        out , _ = self.lstm(x, (h_0.detach(),c_0.detach()))
        out = self.linear(out[:, -1, :])
        
        return out 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = LSTM()
model = model.double() # because we define h_0 and c_0 as double but why we need to define it as double???
model = model.to(device)

loss_f = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr = 1e-3)

print(model)

def train():
    model.train()
    train_loss = 0

    for index,(data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        opt.zero_grad()
        pred = model(data)
        pred = pred.view(-1)

        loss = loss_f(pred,target)
        loss.backward()
        opt.step()

        train_loss += loss

    print("train_loss:{:.6f}".format(train_loss))

    return train_loss


def test():
    model.eval()
    test_loss = 0

    for index,(data,target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        pred = model(data)
        pred = pred.view(-1)

        loss = loss_f(pred,target)
    
        test_loss += loss

    print("test_loss:{:.6f}".format(test_loss))

    return test_loss

def predict(data):
    model.eval()

    with torch.no_grad():
        pred = model(data.to(device))
        return pred

for i in range(100):
    print("epoch = {}".format(i))

    train()
    test()

preds = []

for i in range(len(dataset)):
    normalized_temp , target = dataset[i]

    temp = normalized_temp
    temp = temp.view(1,12,1)

    pred = predict(temp)

    act_pred = scaler.inverse_transform(pred.cpu().reshape(-1,1))
    preds.append(act_pred.item())


months = range(0,df_data.month.size)
mean_temp = df_data.mean_temp

plt.figure(figsize=(15,5))
plt.title("Temp predict",fontsize = 16)
plt.plot(months, mean_temp, 'b', label ='org')
plt.plot(months[12:], preds, 'r', label = 'pred')
plt.legend()
plt.show()


