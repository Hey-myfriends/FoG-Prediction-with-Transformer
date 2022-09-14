import torch
from torch import nn
import pdb

def build_backbone(in_chan, d_model, multilpier=1):
    c = [8, 8, 16, 32, 64, 128, 256, d_model]
    n = [1, 1, 2,  2,  2,  2,   1,   1]
    s = [2, 2, 2,  2,  2,  2,   1,   1]
    return Backbone(in_chan, c, n, s, multiplier=multilpier)

class Backbone(nn.Module):
    def __init__(self, in_chan, c, n, s, multiplier=1):
        super().__init__()
        modulelist = [nn.Sequential(nn.Conv1d(in_chan, int(multiplier*c[0]), 1, stride=s[0]),
            nn.BatchNorm1d(int(multiplier*c[1])))]
        for i in range(1, len(s)-1):
            inc, outc, stride = int(multiplier*c[i-1]), c[i], s[i]
            for j in range(n[i]):
                # print(f"{i+j} input: {inc}, output: {outc}, stride: {stride}")
                modulelist.append(BackboneBase(inc, outc, multiplier, 6, stride=stride))
                inc, stride = outc, 1
                # modulelist.extend([BackboneBase(int(multiplier*c[i-1]), c[i], multiplier, 6, stride=s[i]) for _ in range(n[i])])
        modulelist.append(nn.Sequential(nn.Conv1d(int(multiplier*c[-2]), int(multiplier*c[-1]), 1, stride=s[-1]),
            nn.BatchNorm1d(int(multiplier*c[-1]))))
        self.num_channels = int(multiplier*c[-1])
        self.modulelist = nn.ModuleList(modulelist)

    def forward(self, x):
        # pdb.set_trace()
        for i, m in enumerate(self.modulelist):
            # print(i, x.shape)
            x = m(x)
        return x

class BackboneBase(nn.Module):
    def __init__(self, in_chan, out_chan, multiplier, k, stride=2) -> None:
        super().__init__()

        hidden_dim = k*in_chan
        ## depthwise-conv
        self.DWlayer = self.depthwise_conv(in_chan, hidden_dim)
        self.layer1 = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=stride),
                        nn.BatchNorm1d(hidden_dim), nn.LeakyReLU())

        out_dim = int(out_chan*multiplier)
        self.layer3 = nn.Sequential(nn.Conv1d(hidden_dim, out_dim, 1), nn.BatchNorm1d(out_dim), nn.LeakyReLU())
        ## SE block
        self.SElayer = SEBlock(out_dim, 2)
        self.downsample = nn.Sequential(nn.Conv1d(in_chan, out_dim, 1, stride=stride), nn.BatchNorm1d(int(out_chan*multiplier))) \
            if stride > 1 or in_chan != out_dim else nn.Identity()

    def depthwise_conv(self, i, o):
        return nn.Conv1d(i, o, 3, stride=1, padding=1, groups=i)

    def forward(self, x):
        out = self.DWlayer(x)
        out = self.layer1(out)
        out = self.layer3(out)
        out = self.SElayer(out)
        out += self.downsample(x)
        return out

class SEBlock(nn.Module):
    def __init__(self, in_chan, reduction) -> None:
        super().__init__()
        self.down = nn.Conv1d(in_chan, in_chan//reduction, 1)
        self.relu = nn.LeakyReLU()
        self.up = nn.Conv1d(in_chan//reduction, in_chan, 1)
        self.in_chan = in_chan

    def forward(self, x):
        coef = torch.nn.functional.avg_pool1d(x, kernel_size=x.shape[-1])
        coef = self.down(coef)
        coef = self.relu(coef)
        coef = self.up(coef).sigmoid()
        coef = coef.view(-1, self.in_chan, 1)
        return coef * x

if __name__ == "__main__":
    x = torch.rand(4, 6, 1500)
    # se = SEBlock(32, 2)
    # backbone = BackboneBase(32, 64, 1, 6, 2)
    pdb.set_trace()
    backbone = build_backbone(x.shape[1], 128, 1)
    print(backbone)
    y = backbone(x)
    print(x.shape, y.shape)