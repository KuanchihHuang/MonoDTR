import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import math
import time

from visualDet3D.networks.backbones.dla import dla102
from visualDet3D.networks.backbones.dlaup import DLAUp
from visualDet3D.networks.detectors.dfe import DepthAwareFE
from visualDet3D.networks.detectors.dpe import DepthAwarePosEnc
from visualDet3D.networks.detectors.dtr import DepthAwareTransformer


class MonoDTRCore(nn.Module):
    def __init__(self, backbone_arguments=dict()):
        super(MonoDTRCore, self).__init__()
        self.backbone = dla102(pretrained=True, return_levels=True)
        channels = self.backbone.channels
        self.first_level = 3
        scales = [2**i for i in range(len(channels[self.first_level:]))]
        self.neck = DLAUp(channels[self.first_level:], scales_list=scales)

        self.output_channel_num = 256
        self.dpe = DepthAwarePosEnc(self.output_channel_num)
        self.depth_embed = nn.Embedding(100, self.output_channel_num)
        self.dtr = DepthAwareTransformer(self.output_channel_num)
        self.dfe = DepthAwareFE(self.output_channel_num)
        self.img_conv = nn.Conv2d(self.output_channel_num, self.output_channel_num, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.backbone(x['image'])
        
        x = self.neck(x[self.first_level:])
        
        N, C, H, W = x.shape

        depth, depth_guide, depth_feat = self.dfe(x)
        
        depth_feat = depth_feat.permute(0, 2, 3, 1).view(N, H*W, C)
        
        depth_guide = depth_guide.argmax(1)
        depth_emb = self.depth_embed(depth_guide).view(N, H*W, C)
        depth_emb = self.dpe(depth_emb, (H,W))
        
        img_feat = x + self.img_conv(x)
        img_feat = img_feat.view(N, H*W, C)
        feat = self.dtr(depth_feat, img_feat, depth_emb)
        feat = feat.permute(0, 2, 1).view(N,C,H,W)

        return feat, depth
