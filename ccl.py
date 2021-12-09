import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import os


class CCL(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, two_branch=False, normlinear=False, normalize=False):
        super(CCL, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.two_branch = two_branch
        self.normalize = normalize

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim, two_branch=two_branch, mlp=mlp, normlinear=normlinear)
        self.encoder_k = base_encoder(num_classes=dim, two_branch=two_branch, mlp=mlp, normlinear=normlinear)

        if mlp and not two_branch:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def forward(self, im_q):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        if self.two_branch:
            eq1 = nn.functional.normalize(q[1], dim=1) # branch 2
            q = q[0]                                   # branch 1
        if self.normalize:
            print(1)
            q = nn.functional.normalize(q, dim=1)
        return q



