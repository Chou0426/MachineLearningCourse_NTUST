import torch
import torchvision.datasets as dataset
from model import Generator
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
import torchvision.utils as vutils
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
            transforms.Resize((168, 168)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

root = 'facades'
targetc_path = os.path.join(root, 'custom')
datac_loader = DataLoader(dataset.ImageFolder(targetc_path, transform = transform), batch_size = batch_size, shuffle = True, num_workers = 0)


G_B2A = Generator().to(device)
G_B2A.load_state_dict(torch.load('weights_200_256x256/netG_B2A.pth', map_location = torch.device('cpu')))
G_B2A.eval()

fake_img = []
real_img = []

if __name__ == '__main__':
    progress_bar = tqdm(enumerate(datac_loader), total = len(datac_loader))
    
    for i, data in progress_bar:
        real_image = data[0].to(device)
        fake_image = 0.5 * (G_B2A(real_image).data + 1.0)
        #vutils.save_image(fake_image.detach(), f"output/fake_image_20_lr1e-6_168x168_{i}.jpg", normalize = True)
        fake_img.append(make_grid(fake_image, padding = 0, normalize = True))
        real_img.append(make_grid(real_image, padding = 0, normalize = True))
    fig=plt.figure(figsize=(3,3))
    for i in range(0,3):
        ax_1=fig.add_subplot(3,3,i+1)
        ax_1.imshow(real_img[i].cpu().detach().numpy().transpose(1,2,0))
        # ax_1.imshow(real_img[i].cpu().detach().numpy().transpose(1,2,0))
        ax_1.set_title(str(i+1))
        ax_1.autoscale(enable=True) 
        ax_1.set_xticks([])
        ax_1.set_yticks([])

        ax_2=fig.add_subplot(3,3,i+4)
        ax_2.imshow(fake_img[i].cpu().detach().numpy().transpose(1,2,0))
        # ax_1.imshow(real_img[i].cpu().detach().numpy().transpose(1,2,0))
        ax_2.set_title(str(i+1))
        ax_2.autoscale(enable=True) 
        ax_2.set_xticks([])
        ax_2.set_yticks([])

        ax_3=fig.add_subplot(3,3,i+7)
        ax_3=plt.imread(f"facades/gt/gt_/gt_{i+1}.jpg")
        plt.imshow(ax_3)
        plt.xticks([])
        plt.yticks([])
        plt.title(i+1)

<<<<<<< Updated upstream
    plt.show()
=======
        vutils.save_image(fake_image.detach(), f"output/fake_image_200_256x256.jpg", normalize = True)
>>>>>>> Stashed changes
