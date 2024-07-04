import torch
import torch.nn as nn
import torch.nn.functional as F

class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, act=None, hw=None):
        super(PreActBlock, self).__init__()
        self.act = act
        
        if hw is not None:
            self.ln1 = nn.LayerNorm([in_planes, hw[0], hw[1]])
            hw = [hw[0] // stride, hw[1] // stride]  
            self.ln2 = nn.LayerNorm([planes, hw[0], hw[1]])
        else:
            self.ln1 = nn.BatchNorm2d(in_planes)
            self.ln2 = nn.BatchNorm2d(planes)
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        out = self.act(self.ln1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.act(self.ln2(out)))
        
        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = self.act(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out


class SENet_LN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, act=None, input_size=(32, 32)):
        super(SENet_LN, self).__init__()
        self.act = act
        self.in_planes = 64
        self.input_size = list(input_size) if isinstance(input_size, tuple) else [input_size, input_size]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.ln1 = nn.LayerNorm([64, *self.input_size])
        hw = self.input_size

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, hw=hw)
        hw = [hw[0] // 1, hw[1] // 1] # stride

        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, hw=hw)
        hw = [hw[0] // 2, hw[1] // 2]  

        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, hw=hw)
        hw = [hw[0] // 2, hw[1] // 2]  
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, hw=hw)

        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, hw=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, act=self.act, hw=hw))
            self.in_planes = planes
            if hw is not None:
                hw = [hw[0] // stride, hw[1] // stride]  
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.ln1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SENet18_LN(n_classes, act):
    return SENet_LN(PreActBlock, [2, 2, 2, 2], num_classes=n_classes, act=act)


def test():
    net = SENet18_LN(n_classes=10, act=nn.GELU())
    print(net.layer4)
    y = net(torch.randn(2, 3, 32, 32))
    print(y.size())
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total number of parameters: {total_params:,}")


if __name__ == '__main__':
    test()
