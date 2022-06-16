import torch
import torch.nn as nn

class DepthAwarePosEnc(nn.Module):
    
    def __init__(self, dim, k=3):
        super(DepthAwarePosEnc, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim) 
    
    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == H * W
        feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.view(B,C,-1).transpose(1, 2)

        return x

