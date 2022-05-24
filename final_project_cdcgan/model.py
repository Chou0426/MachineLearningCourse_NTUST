import torch
from torch import nn, optim
import torchvision
from torchsummary import summary


def conv_norm_relu(input_dim, output_dim, kernel_size, stride = 1, padding = 0):
    layer = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding),
                nn.InstanceNorm2d(output_dim),
                nn.ReLU(True)           
    )

    return layer

def dconv_norm_relu(input_dim, output_dim, kernel_size, stride = 1, padding = 0, output_padding = 0):
    layer = nn.Sequential(
                nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding, output_padding),
                nn.InstanceNorm2d(output_dim),
                nn.ReLU(True)
    )
    
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, dim, use_dropout):
        super(ResidualBlock, self).__init__()
        res_block = [nn.ReflectionPad2d(1), # 為了要保留更多特徵值
                    conv_norm_relu(dim, dim, kernel_size = 3)]
        if use_dropout:
            res_block += [nn.Dropout(0.5)]
        
        res_block += [nn.ReflectionPad2d(1),
                    nn.Conv2d(dim, dim, kernel_size = 3),
                    nn.InstanceNorm2d(dim)]

        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        return x + self.res_block(x)

class Generator(nn.Module):
    def __init__(self, input = 3, output = 3, filters = 64, use_dropout = True, n_blocks = 2):
        super(Generator, self).__init__()
        model = [nn.ReflectionPad2d(3),
                conv_norm_relu(input, filters * 1, 7),
                conv_norm_relu(filters * 1, filters * 2, 3, 2, 1),
                conv_norm_relu(filters * 2, filters * 4, 3, 2, 1)]
        for i in range(n_blocks):
            model += [ResidualBlock(filters * 4, use_dropout)]

        model += [dconv_norm_relu(filters * 4, filters * 2, 3, 2, 1, 1),
                dconv_norm_relu(filters * 2, filters * 1, 3, 2, 1, 1),
                nn.ReflectionPad2d(3),
                nn.Conv2d(filters, output, 7),
                nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)      

def conv_norm_leakyrelu(input_dim, output_dim, kernel_size, stride = 1, padding = 0, output_padding = 0):
    layer = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding),
                            nn.InstanceNorm2d(output_dim),
                            nn.LeakyReLU(0.2, True))
    return layer

class Discriminator(nn.Module):
    def __init__(self, input = 3, filters = 64, n_layer = 3):
        super(Discriminator, self).__init__()
        model = [nn.Conv2d(input, filters, kernel_size = 1, stride = 1, padding = 0),
                nn.LeakyReLU(0.2, True)]
        
        for i in range(1, n_layer):
            n_filters_prev = 2 ** (i-1)
            n_filters = 2 ** i
            model += [conv_norm_leakyrelu(filters * n_filters_prev, filters * n_filters, 4, 2, 1)]

        n_filters_prev = 2 ** (n_layer - 1)
        n_filters = 2 ** n_layer
        model += [conv_norm_leakyrelu(filters * n_filters_prev, filters * n_filters, 4, 1, 1)]
        model += [nn.Conv2d(filters * n_filters, 1, 4, 1, 1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class LamdaLR():
    def __init__(self, epochs, offset, decay_epoch):
        self.epoch = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch) / (self.epoch - self.decay_epoch)


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# G = Generator().to(device)
# summary(G, (3, 256, 256))

