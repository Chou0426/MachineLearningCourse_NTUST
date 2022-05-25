import shutil
import torch
import torchvision
from model import Generator, Discriminator, LamdaLR
from torch import nn, optim
import itertools
from itertools import cycle
import random
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt


def weight_init_normal(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)

class ReplayBuffer:
    def __init__(self, max_size = 50):
        assert (max_size > 0), "Empty buffer or trying to create a block hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)

        return torch.cat(to_return)

fake_A_sample = ReplayBuffer()
fake_B_sample = ReplayBuffer()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

D_A = Discriminator().to(device)
D_B = Discriminator().to(device)
G_A2B = Generator().to(device)
G_B2A = Generator().to(device)

D_A.apply(weight_init_normal)
D_B.apply(weight_init_normal)
G_A2B.apply(weight_init_normal)
G_B2A.apply(weight_init_normal)


batch_size = 1
epoch = 20

batch_size = 2
epoch = 200

batch_size = 2
epoch = 200

decay_epoch = 10
lr = 1e-6
log_freq = 100

#loss
MSE = nn.MSELoss()
L1 = nn.L1Loss()

#optim
opt_G = optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr = lr, betas = (0.5, 0.999))
opt_D = optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr = lr, betas = (0.5, 0.999))

lr_scheduler_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda = LamdaLR(epoch, 0, decay_epoch).step)
lr_scheduler_D = optim.lr_scheduler.LambdaLR(opt_D, lr_lambda = LamdaLR(epoch, 0, decay_epoch).step)

#data
transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((286,286)),
            transforms.RandomCrop((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

train_path = 'facades'
trainA_path = os.path.join(train_path, 'trainA')
targetA_path = os.path.join(train_path, 'new_trainA')
trainB_path = os.path.join(train_path, 'trainB')
targetB_path = os.path.join(train_path, 'new_trainB')

if os.path.exists(targetA_path) == False:
    os.makedirs(targetA_path)
    print('create dir:', targetA_path)
    shutil.move(trainA_path, targetA_path)
if os.path.exists(targetB_path) == False:
    os.makedirs(targetB_path)
    print('create dir:', targetB_path)
    shutil.move(trainB_path, targetB_path)

dataA = datasets.ImageFolder(targetA_path, transform = transform)
dataB = datasets.ImageFolder(targetB_path, transform = transform)
dataA_loader = DataLoader(dataA, shuffle = True, batch_size = batch_size, num_workers = 0) # 學姊的是4
dataB_loader = DataLoader(dataB, shuffle = True, batch_size = batch_size, num_workers = 0) 


AVG_G_LOSS = []
AVG_D_LOSS = []

total_len = len(dataA_loader) + len(dataB_loader)

if __name__ == '__main__':
    for i in range(epoch):
        progress_bar = tqdm(enumerate(zip(dataA_loader, dataB_loader)), total = total_len) # 會出現broken pipe問題，是因為windows不能執行num_workers超過0
        G_LOSS = []
        D_LOSS = []
        for idx, data in progress_bar:
            real_A = data[0][0].to(device)
            real_B = data[1][0].to(device)

            #train G
            opt_G.zero_grad()
            fake_A = G_B2A(real_B)
            fake_A_out = D_A(fake_A)
            fake_B = G_A2B(real_A)
            fake_B_out = D_B(fake_B)

            real_label = torch.ones((fake_A_out.size()), dtype = torch.float32).to(device)
            fake_label = torch.zeros((fake_A_out.size()), dtype = torch.float32).to(device)
            adv_loss_B2A = MSE(fake_A_out, real_label)
            adv_loss_A2B = MSE(fake_B_out, real_label)
            adv_loss = adv_loss_A2B + adv_loss_B2A

            rec_A = G_B2A(fake_B)
            rec_B = G_A2B(fake_A)
            consistency_loss_B2A = L1(rec_A, real_A)
            consistency_loss_A2B = L1(rec_B, real_B)
            rec_loss = consistency_loss_A2B + consistency_loss_B2A

            idt_A = G_B2A(real_A)
            idt_B = G_A2B(real_B)
            idt_loss_A = L1(idt_A, real_A)
            idt_loss_B = L1(idt_B, real_B)
            idt_loss = idt_loss_A + idt_loss_B

            #total loss G
            lambda_rec = 10
            lambda_idt = 5
            loss_G = adv_loss + (rec_loss * lambda_rec) + (idt_loss * lambda_idt)

            loss_G.backward()
            opt_G.step()

            

            #train D
            opt_D.zero_grad()
            real_out_A = D_A(real_A)
            real_out_A_loss = MSE(real_out_A, real_label)
            fake_out_A = D_A(fake_A_sample.push_and_pop(fake_A))
            fake_out_A_loss = MSE(fake_out_A, fake_label)
            loss_DA = real_out_A_loss + fake_out_A_loss

            real_out_B = D_B(real_B)
            real_out_B_loss = MSE(real_out_B, real_label)
            fake_out_B = D_B(fake_B_sample.push_and_pop(fake_B))
            fake_out_B_loss = MSE(fake_out_B, fake_label)
            loss_DB = real_out_B_loss + fake_out_B_loss

            loss_D = (loss_DA + loss_DB) * 0.5
            loss_D.backward()
            opt_D.step()

           

            progress_bar.set_description(
                f"[{epoch}/{i}][{idx}/{total_len - 1}]"
                f"Loss_D: {(loss_DA + loss_DB).item():.4f}"
                f"Loss_G: {loss_G.item():4f}"
            )

            if i % log_freq == 0:
                vutils.save_image(real_A, f"output/real_A{epoch}_lr2e-6.jpg", normalize = True)
                vutils.save_image(real_B, f"output/real_B{epoch}_lr2e-6.jpg", normalize = True)
                fake_A = (G_B2A(real_B).data +1.0) * 0.5
                fake_B = (G_B2A(real_A).data +1.0) * 0.5

                vutils.save_image(fake_A, f"output/fake_A{epoch}_lr2e-6.jpg", normalize = True)
                vutils.save_image(fake_B, f"output/fake_B{epoch}_lr2e-6.jpg", normalize = True)

            G_LOSS.append(loss_G.item())
            D_LOSS.append(loss_D.item())


        torch.save(G_A2B.state_dict(), f"weights_200_256x256/netG_A2B_epoch_{epoch}.pth")
        torch.save(G_B2A.state_dict(), f"weights_200_256x256/netG_B2A_epoch_{epoch}.pth")

        lr_scheduler_G.step()
        lr_scheduler_D.step()

        AVG_G_LOSS.append(torch.mean(torch.FloatTensor(G_LOSS)))
        AVG_D_LOSS.append(torch.mean(torch.FloatTensor(D_LOSS)))



    torch.save(G_A2B.state_dict(), f"weights_200_256x256/netG_A2B.pth")
    torch.save(G_B2A.state_dict(), f"weights_200_256x256/netG_B2A.pth")

    plt.figure(1)
    plt.title('LOSS_200_256x256')

    plt.plot(AVG_G_LOSS, 'b', label = 'G_loss')
    plt.plot(AVG_D_LOSS, 'r', label = 'D_loss')
    plt.legend()
    plt.show()


