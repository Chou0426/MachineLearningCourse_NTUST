import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

# Hyperparameter

epoch = 10
batch_size = 64
learning_rate = 1e-4


class Discriminator(nn.Module):
    def __init__(self, c_dim, lable_dim):
        super(Discriminator, self).__init__()
        self.input_x = nn.Sequential(
                        nn.Conv2d(c_dim, 64, 4, 2, 1),
                        nn.LeakyReLU(),
        )
        self.input_y = nn.Sequential(
                        nn.Conv2d(lable_dim, 64, 4, 2, 1),
                        nn.LeakyReLU(),
        )
        self.concat = nn.Sequential(
                        nn.Conv2d(64*2, 64, 4, 2, 1),
                        nn.LeakyReLU(),
                        nn.Conv2d(64, 128, 3, 2, 1),
                        nn.LeakyReLU(),
                        nn.Conv2d(128, 1, 4, 2, 0),
                        nn.Sigmoid(),
        )

    def forward(self, x, y):
        x = self.input_x(x)
        y = self.input_y(y)
        out = torch.cat([x, y], dim = 1)
        out = self.concat(out)

        return out

class Generator(nn.Module):
    def __init__(self, noise_dim, lable_dim):
        super(Generator, self).__init__()
        self.input_x = nn.Sequential(
                        nn.ConvTranspose2d(noise_dim, 256, 4, 1, 0, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(True),
        )
        self.input_y = nn.Sequential(
                        nn.ConvTranspose2d(lable_dim, 256, 4, 1, 0, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(True),
        )
        self.concat = nn.Sequential(
                        nn.ConvTranspose2d(256*2, 128, 4, 2, 1, bias = False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(64, 1, 4, 2, 3, bias = False),
                        nn.Tanh()
        )
    
    def forward(self, x, y):
        x = self.input_x(x)
        y = self.input_y(y)
        out = torch.cat([x, y], dim = 1)
        out = self.concat(out)

        return out

label_dim = 10
z_dim = 100
image_size = 28

# Noise data
temp_noise = torch.randn(label_dim, z_dim) # 產生 10 * 100 維的亂數資料
fixed_noise = temp_noise
fixed_c = torch.zeros(label_dim, 1)  # 產生 10 個種類的對應label

for i in range(0,9):
    fixed_noise = torch.cat([fixed_noise, temp_noise], 0)
    temp = torch.ones(label_dim, 1) + i
    fixed_c = torch.cat([fixed_c, temp], 0)

fixed_noise = fixed_noise.view(-1, z_dim, 1, 1)

# print('pred noise:', fixed_noise.shape)
# print('pred label:', fixed_c.shape, '\t', fixed_c[10])

# One hot
fixed_label = torch.zeros(100, label_dim)
fixed_label.scatter_(1, fixed_c.type(torch.LongTensor), 1)
fixed_label = fixed_label.view(-1, label_dim, 1, 1)
# print('Onehot label:', fixed_label.shape, '\t', fixed_label[10])

# G label
onehot = torch.zeros(label_dim, label_dim)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(label_dim, 1), 1).view(label_dim, label_dim, 1, 1)
# print('Train label G:', onehot.shape, '\n', onehot[8], '\n')

# D label
fill = torch.zeros([label_dim, label_dim, image_size, image_size])
for i in range(label_dim):
    fill[i, i, :, :] = 1

# print('Train D label:', fill.shape, '\t', fill[1])

# Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
train_data = datasets.MNIST('./week13/data/mnist', train = True, transform = transform, download = True)
test_data = datasets.MNIST('./week13/data/mnist', train = False, transform = transform, download = True)

# Create DataLoader
train_loader = DataLoader(train_data, shuffle = True, batch_size = batch_size, drop_last = True)
test_loader = DataLoader(test_data, shuffle = False, batch_size = batch_size)

# Setting optimizer, device, loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D = Discriminator(1, 10).to(device)
G = Generator(100, 10).to(device)
loss_f = nn.BCELoss()
opt_D = optim.Adam(D.parameters(), lr = learning_rate)
opt_G = optim.Adam(G.parameters(), lr = learning_rate)

D_avgloss = []
G_avgloss = []

for i in range(epoch):
    D_loss = []
    G_loss = []

    for idx, (data, target) in enumerate(train_loader):
        # train D
        opt_D.zero_grad()
        x_real = data.to(device)
        label = target.to(device)
        y_real = torch.ones(batch_size,).to(device)
        c_real = fill[label].to(device)
        y_real_pred = D(x_real, c_real).squeeze()
        d_real_loss = loss_f(y_real_pred, y_real)
        d_real_loss.backward()

        noise = torch.randn(batch_size, z_dim, 1, 1, device = device)
        noise_label = (torch.rand(batch_size, 1) * label_dim).type(torch.LongTensor).squeeze()
        noise_label_onehot = onehot[noise_label].to(device)
        x_fake = G(noise, noise_label_onehot)
        y_fake = torch.zeros(batch_size,).to(device)
        c_fake = fill[noise_label].to(device)
        y_fake_pred = D(x_fake, c_fake).squeeze()
        d_fake_loss = loss_f(y_fake_pred, y_fake)
        d_fake_loss.backward()
        opt_D.step()

        # train G
        opt_G.zero_grad()
        noise = torch.randn(batch_size, z_dim, 1, 1, device = device)
        noise_label = (torch.rand(batch_size, 1) * label_dim).type(torch.LongTensor).squeeze()
        noise_label_onehot = onehot[noise_label].to(device)
        x_fake = G(noise, noise_label_onehot)
        c_fake = fill[noise_label].to(device)
        y_fake_pred = D(x_fake, c_fake).squeeze()
        g_loss = loss_f(y_fake_pred, y_real)
        g_loss.backward()
        opt_G.step()

        D_loss.append(d_fake_loss.item() + d_real_loss.item())
        G_loss.append(g_loss.item())

        if idx % (int(len(train_loader)/100)) == 0:
            with torch.no_grad():
                print("Epoch[{:02}/{}] \t step[{:05}/{}] \t D_loss:{:.6} \t G_loss:{:.6}".format(i+1, epoch, idx+1, len(train_loader), D_loss[idx], G_loss[idx]))

    D_avgloss.append(torch.mean(torch.FloatTensor(D_loss)))
    G_avgloss.append(torch.mean(torch.FloatTensor(G_loss)))


torch.save(G.state_dict(),'pt/cGenerator.pt')

plt.figure(1)
plt.plot(D_avgloss, label = 'D_loss')
plt.plot(G_avgloss, label = 'G_loss')
plt.legend()
plt.show()
