'''ShuffleNetV2 in PyTorch.

See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


class BasicBlock(nn.Module):
    def __init__(self, in_channels, split_ratio=0.5,act=nn.ReLU()):
        super(BasicBlock, self).__init__()
        self.act = act
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = self.act(self.bn1(self.conv1(x2)))
        out = self.bn2(self.conv2(out))
        out = self.act(self.bn3(self.conv3(out)))
        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels,act):
        super(DownBlock, self).__init__()
        self.act = act
        mid_channels = out_channels // 2
        # left
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # right
        self.conv3 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv4 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_channels)
        self.conv5 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_channels)

        self.shuffle = ShuffleBlock()

    def forward(self, x):
        # left
        out1 = self.bn1(self.conv1(x))
        out1 = self.act(self.bn2(self.conv2(out1)))
        # right
        out2 = self.act(self.bn3(self.conv3(x)))
        out2 = self.bn4(self.conv4(out2))
        out2 = self.act(self.bn5(self.conv5(out2)))
        # concat
        out = torch.cat([out1, out2], 1)
        out = self.shuffle(out)
        return out


configs_cifar = {
    0.5: {
        'out_channels': (48, 96, 192, 1024),
        'num_blocks': (3, 7, 3)
    },

    1.: {
        'out_channels': (116, 232, 464, 1024),
        'num_blocks': (3, 7, 3)
    },
    1.5: {
        'out_channels': (176, 352, 704, 1024),
        'num_blocks': (3, 7, 3)
    },
    2.: {
        'out_channels': (224, 488, 976, 2048),
        'num_blocks': (3, 7, 3)
    }
}

class ShuffleNetV2(nn.Module):
    def __init__(self, net_size,act,n_classes,configs=configs_cifar,img_size=32):
        super(ShuffleNetV2, self).__init__()
        self.img_size = img_size 
        self.out_channels = configs[net_size]['out_channels']
        num_blocks = configs[net_size]['num_blocks']
        self.act = act

        self.conv1 = nn.Conv2d(3, 24, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_channels = 24
        self.layer1 = self._make_layer(self.out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(self.out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(self.out_channels[2], num_blocks[2])
        self.conv2 = nn.Conv2d(self.out_channels[2], self.out_channels[3],
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels[3])
        self.linear = nn.Linear(self.out_channels[3], n_classes)

    def _make_layer(self, out_channels, num_blocks):
        layers = [DownBlock(self.in_channels, out_channels,self.act)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels,act=self.act))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        # out = F.max_pool2d(out, 3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.act(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out,4 if self.img_size==64 else 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def shufflenet_tiny(net_size,act,n_classes):
    configs_tiny = {
        0.5: {
            'out_channels': (48, 96, 192, 1024),
            'num_blocks': (3, 7, 3)
        },

        1.: {
            'out_channels': (116, 232, 464, 1024),
            'num_blocks': (3, 7, 3)
        },
        1.5: {
            'out_channels': (176, 352, 704, 1024),
            'num_blocks': (3, 7, 3)
        },
        2.: {
            'out_channels': (224, 488, 976, 1024),
            'num_blocks': (3, 7, 3)
        }
    }

    return ShuffleNetV2(net_size=net_size,act=act,n_classes=n_classes,configs=configs_tiny,img_size=64)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def test():
    act= nn.ReLU()
    net = ShuffleNetV2(net_size=1.,n_classes=100,act=act)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.shape)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"1.x Total number of parameters: {total_params:,}")

    net = ShuffleNetV2(net_size=2.,n_classes=100,act=act)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.shape)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"2.x Total number of parameters: {total_params:,}")


if __name__ == "__main__":
    test()
