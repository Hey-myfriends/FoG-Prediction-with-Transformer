from transforms3d.axangles import axangle2mat
import numpy as np
from scipy.interpolate import CubicSpline
import torch

def DA_Rotation(data: torch.Tensor):
    assert data.shape[1] % 3 == 0, "The channel dim must be in dim 1 and dividable by 3."
    axis = np.random.uniform(low=-1, high=1, size=3)
    angle = np.random.uniform(low=-np.pi, high=np.pi, size=1)
    W = axangle2mat(axis, angle)
    W = torch.Tensor(W)
    for i in range(data.shape[1] // 3):
        data[:, i*3:(i+1)*3] = data[:, i*3:(i+1)*3] @ W
    return data