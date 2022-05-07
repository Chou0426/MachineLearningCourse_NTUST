import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import time

batch_size = 64
epoch = 100

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
                    nn.Conv2d(1, 64, 4, 2, 1),
                    nn.LeakyReLU(),
                    nn.Conv2d(64, 128, 4, 2, 1),
                    nn.LeakyReLU(),
                    nn.Conv2d(128, 256, 3, 2, 1),
                    nn.LeakyReLU(),
                    nn.Conv2d(256, 1, 4, 2, 0),
                    nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)

        return x


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
                    nn.ConvTranspose2d(z_dim, 256, 4, 1, 0, bias = False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(128, 64, 4, 2, 1, bias= False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(64, 1, 4, 2, 3, bias = False),
                    nn.Tanh()
        )

    def forward(self,x):
        x = self.net(x)

        return x

'''Load Mnist Dataset'''

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ),(0.5, ))])

train_data = datasets.MNIST('data/mnist', train = True, transform = transform, download = True)
test_data = datasets.MNIST('data/mnist', train = False, transform = transform, download = True)

train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last = True)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

D = Discriminator().to(device)
G = Generator(100).to(device)

loss_f = nn.BCELoss()
D_opt = optim.Adam(D.parameters(), lr = 1e-5)
G_opt = optim.Adam(G.parameters(), lr = 1e-5)


D_loss = []
G_loss = []

start = time.time()
print("The time used to execute this is given below")


for i in range(0, epoch):
    for index, (data, target) in enumerate(train_loader):
        # print(index) # 0~936
        # print(data) # 64

        # Training D
        D_opt.zero_grad()
        x_real = data.to(device)
        y_real = torch.ones(batch_size, ).to(device)
        y_real_pred = D(x_real)
        d_real_loss = loss_f(y_real_pred.view(-1), y_real)
        d_real_loss.backward()

        noise = torch.randn(batch_size, 100, 1, 1, device = device)
        x_fake = G(noise)
        y_fake = torch.zeros(batch_size, ).to(device)
        y_fake_pred = D(x_fake)
        d_fake_loss = loss_f(y_fake_pred.view(-1), y_fake)
        d_fake_loss.backward()

        D_loss.append(d_fake_loss.item() + d_real_loss.item())
        D_opt.step()

        # Training G
        G_opt.zero_grad()
        noise = torch.randn(batch_size, 100, 1, 1, device = device)
        x_fake = G(noise)
        fake = torch.ones(batch_size, ).to(device)
        fake_pred = D(x_fake)
        g_loss = loss_f(fake_pred.view(-1), fake)
        g_loss.backward()

        G_loss.append(g_loss)
        G_opt.step()

        # print loss per 100 iterations
        if index % 100 == 0:
            with torch.no_grad():
                print("[{},{}], D_loss : {:.6f}, G_loss : {:.6f}".format(i+1, 1, D_loss[index], G_loss[index]))

end = time.time()
print(end - start)

torch.save(G.state_dict(), f'Mnist_Generator_epoch{epoch}.pt')

plt.figure(1)
plt.title(f'Mnist_loss_epoch{epoch}')
plt.plot(G_loss, 'b', label = 'G_loss')
plt.plot(D_loss, 'r', label = 'D_loss')
plt.legend()
plt.savefig(f'Mnist_loss{epoch}.png')
plt.show()



