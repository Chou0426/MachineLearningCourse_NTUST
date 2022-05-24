import torch
import torchvision.datasets as dataset
from model import Generator
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
import torchvision.utils as vutils

batch_size = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

root = 'facades'
targetc_path = os.path.join(root, 'custom')
datac_loader = DataLoader(dataset.ImageFolder(targetc_path, transform = transform), batch_size = batch_size, shuffle = True, num_workers = 0)

G_B2A = Generator().to(device)
G_B2A.load_state_dict(torch.load('weights_5/netG_B2A.pth', map_location = torch.device('cpu')))
G_B2A.eval()

if __name__ == '__main__':
    progress_bar = tqdm(enumerate(datac_loader), total = len(datac_loader))
    for i, data in progress_bar:
        real_image = data[0].to(device)
        fake_image = 0.5 * (G_B2A(real_image).data + 1.0)

        vutils.save_image(fake_image.detach(), f"output/fake_image_5.jpg", normalize = True)