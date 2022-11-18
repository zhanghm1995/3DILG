import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DisPU_Generator(nn.Module):
    def __init__(self, N=256, K=1024, dim=256, M=1024, num_neighbors=32, is_training=True, **kwargs):
        super().__init__()

        self.is_training = is_training
        self.reuse = False
        self.up_ratio = M / N
        self.num_point = N  # 256
        self.out_num_point = M # 1024

    def forward(self, pc, bins=256):
        # pc: B x N x D
        B, N, D = pc.shape
        assert N == self.out_num_point

        use_bn = False
        n_layer = 6
        K = 16
        filter = 24
        use_noise = False
        dense_block = 4
        bn_decay = 0.95
        use_sm = True
        step_ratio = self.up_ratio
        fine_extracotr = False
        is_off = True
        refine = True

        coarse_feat = ops.feature_extraction_GCN(pc, scope='feature_extraction_coarse', dense_block=dense_block,
                                                 growth_rate=filter, is_training=self.is_training,
                                                 bn_decay=bn_decay, use_bn=use_bn)  ## [B.N.C]




        flattened = pc.view(B * N, D)

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)