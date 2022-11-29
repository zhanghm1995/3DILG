from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math
from PointAttN_utils.model_utils import *
from pointnet2.utils import pointnet2_utils as pn2_utils
import point_operation
from PUCRN.PUCRN_new import MLP_CONV

# from utils.mm3d_pn2 import furthest_point_sample, gather_points
from torch.nn import Sequential as Seq
from torch.nn import Linear as Lin
import numpy as np
class cross_transformer(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    # 原始的transformer
    def forward(self, src1, src2, embed1, embed2, if_act=False):
        src1 = self.with_pos_embed(src1, pos=embed1)
        src2 = self.with_pos_embed(src2, pos=embed2)
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)

        src1 = src1.permute(1, 2, 0)

        return src1


class PCT_refine(nn.Module):
    def __init__(self, channel=128, ratio=1):
        super(PCT_refine, self).__init__()
        self.ratio = ratio
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)

        self.sa1 = cross_transformer(channel, channel)
        self.sa2 = cross_transformer(channel, channel)
        self.sa3 = cross_transformer(channel, channel * ratio)

        self.relu = nn.GELU()

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)

        self.channel = channel

        self.conv_delta = nn.Conv1d(channel * 2, channel * 1, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel * ratio, channel * ratio, kernel_size=1)

        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)

        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)

        self.regressor = MLP_CONV(in_channel=256, layer_dims=[128, 64, 3])

    def forward(self, feature, xyz, embedding):
        '''
        :param x: input feature
        :param coarse: input xyz
        :return:
        '''
        xyz = xyz.permute(0, 2, 1).contiguous()
        batch_size, _, N = xyz.size()

        # y = self.conv_x1(self.relu(self.conv_x(xyz)))  # B, C, N     3-> 256

        # feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))  # B, C, N
        # y0 = y #torch.cat([y, feat_g.repeat(1, 1, y.shape[-1])], dim=1)
        y0 = feature
        # print('==========================', y0.size(), embedding.size())
        y1 = self.sa1(y0, y0, embed1=embedding, embed2=embedding)
        y2 = self.sa2(y1, y1, embed1=embedding, embed2=embedding)
        y3 = self.sa3(y2, y2, embed1=embedding, embed2=embedding)
        y3 = self.conv_ps(y3).reshape(batch_size, -1, N * self.ratio)  # [B,256,1024]
        up_point = self.regressor(y3)
        # predict = up_point + xyz.repeat(1, 1, self.ratio)
        # y_up = y.repeat(1, 1, self.ratio)
        # y_cat = torch.cat([y3, y_up], dim=1)
        # y4 = self.conv_delta(y_cat)
        #
        # x = self.conv_out(self.relu(self.conv_out1(y4))) + xyz.repeat(1, 1, self.ratio)
        x = up_point.float().permute(0, 2, 1).contiguous() #x.
        return x  # final xyz, final feature

def embed(input, basis):
    # print(input.size(), basis.size())
    projections = torch.einsum(
        'bnd,de->bne', input, basis)  # .permute(2, 0, 1)
    # print(projections.max(), projections.min())
    # print(projections.size()) #torch.Size([B, 512, 24])
    embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
    return embeddings  # B x N x E

class PCT_encoder(nn.Module):
    def __init__(self, channel=64):
        super(PCT_encoder, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)

        self.sa1 = cross_transformer(channel, channel)
        self.sa1_1 = cross_transformer(channel * 2, channel * 2)
        self.sa2 = cross_transformer((channel) * 2, channel * 2)
        self.sa2_1 = cross_transformer((channel) * 4, channel * 4)
        self.sa3 = cross_transformer((channel) * 4, channel * 4)
        self.sa3_1 = cross_transformer((channel) * 8, channel * 8)

        self.relu = nn.GELU()

        # self.sa0_d = cross_transformer(channel * 8, channel * 8)
        # self.sa1_d = cross_transformer(channel * 8, channel * 8)
        # self.sa2_d = cross_transformer(channel * 8, channel * 8)

        self.conv_out = nn.Conv1d(512, 256, kernel_size=1)
        # self.conv_out1 = nn.Conv1d(channel * 4, 64, kernel_size=1)
        # self.ps = nn.ConvTranspose1d(channel * 8, channel, 128, bias=True)
        # self.ps_refuse = nn.Conv1d(channel, channel * 8, kernel_size=1)
        # self.ps_adj = nn.Conv1d(channel * 8, channel * 8, kernel_size=1)

        #positional embedding
        self.embed_1 = Seq(Lin(48 + 3, channel * 2))  # , nn.GELU(), Lin(128, 128))
        self.embed_2 = Seq(Lin(48 + 3, channel * 4))
        self.embed_3 = Seq(Lin(48 + 3, channel * 8))

        self.embedding_dim = 48
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16



    def original_forward(self, points):
        batch_size, _, N = points.size()

        x = self.relu(self.conv1(points))  # B, D, N
        x0 = self.conv2(x)

        # GDP
        idx_0 = pn2_utils.furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)
        x_g0 = pn2_utils.gather_operation(x0, idx_0)
        points = pn2_utils.gather_operation(points, idx_0)
        x1 = self.sa1(x_g0, x0).contiguous()
        x1 = torch.cat([x_g0, x1], dim=1)
        # SFA
        x1 = self.sa1_1(x1, x1).contiguous()
        # GDP
        idx_1 = pn2_utils.furthest_point_sample(points.transpose(1, 2).contiguous(), N // 8)
        x_g1 = pn2_utils.gather_operation(x1, idx_1)
        points = pn2_utils.gather_operation(points, idx_1)
        x2 = self.sa2(x_g1, x1).contiguous()  # C*2, N
        x2 = torch.cat([x_g1, x2], dim=1)
        # SFA
        x2 = self.sa2_1(x2, x2).contiguous()

        # GDP
        idx_2 = pn2_utils.furthest_point_sample(points.transpose(1, 2).contiguous(), N // 16)
        x_g2 = pn2_utils.gather_operation(x2, idx_2)
        # points = pn2_utils.gather_operation(points, idx_2)
        x3 = self.sa3(x_g2, x2).contiguous()  # C*4, N/4
        x3 = torch.cat([x_g2, x3], dim=1)
        # SFA
        x3 = self.sa3_1(x3, x3).contiguous()

        # seed generator
        # maxpooling
        x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)
        x = self.relu(self.ps_adj(x_g))
        x = self.relu(self.ps(x))
        x = self.relu(self.ps_refuse(x))
        # SFA
        x0_d = (self.sa0_d(x, x))
        x1_d = (self.sa1_d(x0_d, x0_d))
        x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size, self.channel * 4, N // 8)

        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))

        return x_g, fine

    def forward(self, ori_points):
        '''
        input ori_points: torch.Size([B, 1024, 3])
        return  sampled pc feature: torch.Size([B, N=128, C=256]); sampled pc coord: torch.Size([64, 128, 3])
        '''
        points = ori_points.permute(0, 2, 1).contiguous()

        batch_size, D, N = points.size()
        flattened = ori_points.view(batch_size * N, D)

        x = self.relu(self.conv1(points))  # B, D, N
        x0 = self.conv2(x).float()

        # GDP
        # idx_0 = pn2_utils.furthest_point_sample(ori_points, N // 2)
        num_needed = int(N // 2)  # 16 times  * self.ratio
        idx_0 = flattened.new_zeros((batch_size, num_needed), dtype=torch.int)
        for i in range(batch_size):
            idx_0[i] = torch.randperm(N, device=points.device)[:num_needed]

        x_g0 = pn2_utils.gather_operation(x0, idx_0)
        points = pn2_utils.gather_operation(points, idx_0)  # sampled pc coordinate
        xyz = points.permute(0, 2, 1).contiguous()
        embeddings_1 = embed(xyz, self.basis)
        # print('==============',xyz.size(), embeddings_1.size())
        embeddings_1 = self.embed_1(torch.cat([xyz, embeddings_1], dim=2))  # 3 + 48 -> 256

        x1 = self.sa1(x_g0, x0, embed1=embeddings_1, embed2=embeddings_1).contiguous()
        x1 = torch.cat([x_g0, x1], dim=1)
        # SFA
        x1 = self.sa1_1(x1, x1, embed1=embeddings_1, embed2=embeddings_1).contiguous().float()

        # GDP
        num_needed = int(N // 4)  # 16 times  * self.ratio
        idx_1 = flattened.new_zeros((batch_size, num_needed), dtype=torch.int)
        for i in range(batch_size):
            idx_1[i] = torch.randperm(N // 2, device=points.device)[:num_needed]

        # idx_1 = pn2_utils.furthest_point_sample(points.permute(0, 2, 1).contiguous(), N // 4)
        x_g1 = pn2_utils.gather_operation(x1.contiguous(), idx_1)  # B,C,N
        points = pn2_utils.gather_operation(points, idx_1)
        xyz = points.permute(0, 2, 1).contiguous()
        embeddings_2 = embed(xyz, self.basis)
        embeddings_2 = self.embed_2(torch.cat([xyz, embeddings_2], dim=2))  # 3 + 48 -> 256
        # x_g1 = pn2_utils.gather_operation(x1, idx_1)
        # points = pn2_utils.gather_operation(points, idx_1)
        x2 = self.sa2(x_g1, x1, embed1=embeddings_2, embed2=embeddings_1).contiguous()  # C*2, N
        x2 = torch.cat([x_g1, x2], dim=1)
        # SFA
        x2 = self.sa2_1(x2, x2, embed1=embeddings_2, embed2=embeddings_2).contiguous().float()

        # GDP
        # idx_2 = pn2_utils.furthest_point_sample(points.permute(0, 2, 1).contiguous(), N // 8)
        num_needed = int(N // 8)  # 16 times  * self.ratio
        idx_2 = flattened.new_zeros((batch_size, num_needed), dtype=torch.int)
        for i in range(batch_size):
            idx_2[i] = torch.randperm(N // 4, device=points.device)[:num_needed]
        x_g2 = pn2_utils.gather_operation(x2, idx_2)  # B,C,N
        # x_g2 = pn2_utils.gather_operation(x2, idx_2)
        points = pn2_utils.gather_operation(points, idx_2)
        xyz = points.permute(0, 2, 1).contiguous()
        embeddings_3 = embed(xyz, self.basis)
        embeddings_3 = self.embed_3(torch.cat([xyz, embeddings_3], dim=2))  # 3 + 48 -> 256
        x3 = self.sa3(x_g2, x2, embed1=embeddings_3, embed2=embeddings_2).contiguous()  # C*4, N/4
        x3 = torch.cat([x_g2, x3], dim=1)
        # SFA
        x3 = self.sa3_1(x3, x3, embed1=embeddings_3, embed2=embeddings_3).contiguous()

        # # seed generator
        # # maxpooling
        # x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)
        # x = self.relu(self.ps_adj(x_g))
        # x = self.relu(self.ps(x))
        # x = self.relu(self.ps_refuse(x))
        # # SFA
        # x0_d = (self.sa0_d(x, x))
        # x1_d = (self.sa1_d(x0_d, x0_d))
        # x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size, self.channel * 4, N // 8)
        #
        # fine = self.conv_out(self.relu(self.conv_out1(x2_d)))
        sparse_feature = self.conv_out(x3)
        # print('=================output feature size: ', sparse_feature.size())  # torch.Size([B, C=256, N=128])
        sparse_feature = sparse_feature.permute(0, 2, 1).contiguous()
        points = points.permute(0, 2, 1).contiguous()
        return sparse_feature, points
