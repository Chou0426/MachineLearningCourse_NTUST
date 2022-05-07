import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


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

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

G = Generator(100).to(device)
G.load_state_dict(torch.load('Mnist_Generator_epoch10.pt', map_location = torch.device('cpu')))

img = []
noise = torch.randn(32, 100, 1, 1, device = device) # 32張
fake = G(noise)
img.append(make_grid(fake, padding = 0, normalize = True)) 
im = img[0].cpu().detach().numpy().transpose(1,2,0)  # transpose(1,2,0):調換位置 -> 1:img_size, 2:img_size, 0:channel_size
plt.imshow(im)
plt.xticks([]) # 將圖片的刻度移除
plt.yticks([])
plt.savefig('Mnist_Result10.png')
plt.show()

