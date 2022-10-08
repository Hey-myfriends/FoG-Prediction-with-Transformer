
import numpy as np
import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F
torch.manual_seed(10)
torch.randperm(10)
np.random.seed(10)

class SEC_ALSTM(nn.Module):
    '''
    reproduced from paper "Improved Deep leanring Technique to Detect freezing of gait in Parkinson's Disease based on wearbale sensors"
    '''
    def __init__(self, in_c, out_c, num_classes=3, hid_rnn=128, r=1): # [N, 6, 1500]
        super(SEC_ALSTM, self).__init__()

        # layer 1
        self.layer1 = nn.Sequential(nn.Conv1d(in_c, out_c, kernel_size=7, padding=3),
                                    nn.BatchNorm1d(out_c), nn.ReLU(), nn.MaxPool1d(2)) # [N, 3, 750]

        # layer 2
        self.SE2 = SE_blk(out_c, r=r) # [N, 3, 750]

        # layer 3
        self.layer3 = nn.Sequential(nn.Conv1d(out_c, out_c, kernel_size=5, padding=2),
                                    nn.BatchNorm1d(out_c), nn.ReLU(), nn.MaxPool1d(2)) # [N, 3, 375]

        # layer 4
        self.SE4 = SE_blk(out_c, r=r) # [N, 3, 375]

        # layer 5
        self.layer5 = nn.Sequential(nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
                                    nn.BatchNorm1d(out_c), nn.ReLU()) # [N, 3, 375]

        # layer 6
        self.ALSTM = Attn_LSTM(out_c, hid_rnn) # [N, hid_rnn]

        # layer 7
        self.layer7 = nn.Linear(hid_rnn, num_classes)

        self.init_net()

    def init_net(self):
        initrange = 0.1
        self.layer1[0].weight.data.uniform_(-initrange, initrange)
        self.layer1[0].bias.data.zero_()
        self.layer3[0].weight.data.uniform_(-initrange, initrange)
        self.layer3[0].bias.data.zero_()
        self.layer5[0].weight.data.uniform_(-initrange, initrange)
        self.layer5[0].bias.data.zero_()
        self.layer7.weight.data.uniform_(-initrange, initrange)
        self.layer7.bias.data.zero_()

    def forward(self, x):
        # breakpoint()
        x = self.layer1(x)
        x = self.SE2(x)
        x = self.layer3(x)
        x = self.SE4(x)
        x = self.layer5(x)
        x = self.ALSTM(x)
        x = self.layer7(x)
        return {"pred_logits": x}

class SE_blk(nn.Module): # squeeze and excitation block
    def __init__(self, in_c, r=2):
        super(SE_blk, self).__init__()
        self.GAP = nn.AdaptiveAvgPool1d(1) # squeeze
        self.excite = nn.Sequential(nn.Linear(in_c, in_c//r), nn.ReLU(),
                                    nn.Linear(in_c//r, in_c), nn.Sigmoid())
        self.init_net()

    def init_net(self):
        initrange = 0.1
        self.excite[0].weight.data.uniform_(-initrange, initrange)
        self.excite[0].bias.data.zero_()
        self.excite[2].weight.data.uniform_(-initrange, initrange)
        self.excite[2].bias.data.zero_()

    def forward(self, x): # [N, C, H_in]
        z = self.GAP(x) # [N, C, 1]
        z = z.squeeze(-1) # [N, C]
        W_scale = self.excite(z).unsqueeze(-1) # [N, C, 1]
        x = torch.mul(W_scale, x)
        return x

class Attn_LSTM(nn.Module):
    def __init__(self, in_size, hidd_size):
        super(Attn_LSTM, self).__init__()
        self.hidd_size = hidd_size
        self.lstm = nn.LSTM(in_size, hidd_size, batch_first=False)
        self.l1 = nn.Linear(hidd_size, hidd_size)
        self.smx = nn.Softmax(dim=1)
        self.init_net()

    def init_net(self):
        initrange = 0.1
        self.l1.weight.data.uniform_(-initrange, initrange)
        self.l1.bias.data.zero_()

    def forward(self, x):
        # breakpoint()
        x = x.permute(2, 0, 1) #input shape: [L, N, H_in]
        output, (h_n, _) = self.lstm(x) # hidden state vectors are the input of attn layer
        # print('h_n[0] == output[-1] -> ', torch.all(h_n[-1] == output[-1]).item())
        # output : [L, N, H_out], Hn : [nulayers, N, H_out]
        output = self.l1(output)
        W_T = self.smx(torch.bmm(output.permute(1,0,2), h_n[-1].unsqueeze(-1))) # [N, L, 1]
        output = torch.bmm(output.permute(1,2, 0), W_T) # [N, H_out, 1]
        return output.squeeze(-1) # [N, H_out]

if __name__ == "__main__":
    x = torch.rand(32, 6, 1500)
    breakpoint()
    model = SEC_ALSTM(6, 32, num_classes=3)
    y = model(x)
    print(y.shape)