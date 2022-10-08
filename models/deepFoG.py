# from FoG_prediction-new.summary import output
from turtle import forward
import torch
from torch import nn

def net(in_chans):
    return nn.Sequential(nn.Conv1d(in_chans, 100, kernel_size=10), nn.MaxPool1d(kernel_size=3),
            nn.AvgPool1d(kernel_size=2), nn.Conv1d(100, 40, kernel_size=10), nn.AdaptiveAvgPool1d(output_size=1), 
            nn.Dropout(0.5), nn.Flatten(start_dim=1), nn.Linear(40, 3))

class deepFoG(nn.Module):
    def __init__(self, in_chans) -> None:
        super().__init__()
        self.inchan = in_chans
        self.net = net(in_chans)

    def forward(self, x):
        out = self.net(x)
        return {"pred_logits": out}

if __name__ == "__main__":
    x = torch.rand(16, 6, 1500)
    m = deepFoG(x.shape[1])
    y = m(x)
    print(y["pred_logits"].shape)