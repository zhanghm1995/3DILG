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
from quantize import EMAVectorQuantizer
# from modeling_vqvae import VectorQuantizer2
from torch import einsum
from einops import rearrange


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
    def forward(self, src1, src2, embed1=None, embed2=None, if_act=False):
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


class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta=0.25, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.embedding.weight.data.normal_(0, 0.01)  # 1

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c n -> b n c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        print(torch.unique(min_encoding_indices))
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # print(torch.mean(z_q**2), torch.mean(z**2))

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b n c -> b c n').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, perplexity, min_encoding_indices.view(z.shape[0], z.shape[1])

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:  # B x C x N
            z_q = z_q.view(shape)
            # # reshape back to match original input shape
            z_q = z_q.permute(0, 2, 1).contiguous()

        return z_q


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

        # self.codebook = VectorQuantizer2(1024, 256)  # EMAVectorQuantizer(1024, 256, beta=0.25)

    def forward(self, feature, xyz, embedding):
        '''
        :param x: input feature
        :param coarse: input xyz
        :param embedding: positional embedding
        :return:
        '''
        xyz = xyz.permute(0, 2, 1).contiguous()
        batch_size, _, N = xyz.size()

        # y = self.conv_x1(self.relu(self.conv_x(xyz)))  # B, C, N     3-> 256

        # feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))  # B, C, N
        # y0 = y #torch.cat([y, feat_g.repeat(1, 1, y.shape[-1])], dim=1)
        y0 = feature  # torch.Size([64, C=256, N=128])
        # print('==========================', y0.size(), embedding.size())
        y1 = self.sa1(y0, y0, embedding, embedding)
        y2 = self.sa2(y1, y1, embedding, embedding)  #
        y3 = self.sa3(y2, y2, embedding, embedding)  #
        y3 = self.conv_ps(y3).reshape(batch_size, -1, N * self.ratio)  # [B,256,1024]
        # y3_q, loss_vq, perplexity, min_encoding_indices = self.codebook(y3)  # B C rN
        # perplexity, encodings, encoding_indices = info
        up_point = self.regressor(y3)

        # TODO: VQ on up_point

        predict = up_point + xyz.repeat(1, 1, self.ratio)
        # y_up = y.repeat(1, 1, self.ratio)
        # y_cat = torch.cat([y3, y_up], dim=1)
        # y4 = self.conv_delta(y_cat)
        #
        # x = self.conv_out(self.relu(self.conv_out1(y4))) + xyz.repeat(1, 1, self.ratio)
        x = predict.float().permute(0, 2, 1).contiguous()
        return x #, loss_vq  # final xyz, final feature


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
        # self.sa3 = cross_transformer((channel) * 4, channel * 4)
        # self.sa3_1 = cross_transformer((channel) * 8, channel * 8)

        self.relu = nn.GELU()

        # self.sa0_d = cross_transformer(channel * 8, channel * 8)
        # self.sa1_d = cross_transformer(channel * 8, channel * 8)
        # self.sa2_d = cross_transformer(channel * 8, channel * 8)

        self.conv_out = nn.Conv1d(channel * 4, 256, kernel_size=1)
        # self.conv_out1 = nn.Conv1d(channel * 4, 64, kernel_size=1)
        # self.ps = nn.ConvTranspose1d(channel * 8, channel, 128, bias=True)
        # self.ps_refuse = nn.Conv1d(channel, channel * 8, kernel_size=1)
        # self.ps_adj = nn.Conv1d(channel * 8, channel * 8, kernel_size=1)

        # positional embedding
        self.embed_1 = Seq(Lin(48 + 3, channel))  # , nn.GELU(), Lin(128, 128))
        # self.embed_2 = Seq(Lin(48 + 3, channel * 4))
        # self.embed_3 = Seq(Lin(48 + 3, channel * 8))

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
        xyz = points.permute(0, 2, 1).contiguous()  # torch.Size([64, 1024, 3])
        flattened = ori_points.view(batch_size * N, D)

        x = self.relu(self.conv1(points))  # B, D, N
        x0 = self.conv2(x).float()  # torch.Size([B=64, 64, 1024])

        # GDP
        # idx_0 = pn2_utils.furthest_point_sample(ori_points, N // 2)
        num_needed = int(N // 2)  # 16 times  * self.ratio
        idx_0 = flattened.new_zeros((batch_size, num_needed), dtype=torch.int)
        for i in range(batch_size):
            idx_0[i] = torch.randperm(N, device=points.device)[:num_needed]

        x_g0 = pn2_utils.gather_operation(x0, idx_0)  # torch.Size([B=64, 64, 512])
        points = pn2_utils.gather_operation(points, idx_0)  # sampled pc coordinate

        embeddings_x0 = embed(xyz, self.basis)  # torch.Size([64, 1024, 48])
        embeddings_x0 = self.embed_1(torch.cat([xyz, embeddings_x0], dim=2)).float().permute(0, 2,
                                                                                             1).contiguous()  # 3 + 48 -> 256 64

        embedding_g0 = pn2_utils.gather_operation(embeddings_x0, idx_0)
        # print(x_g0.size(), x0.size(), embedding_g0.size(), embeddings_x0.size())
        x1 = self.sa1(x_g0, x0, embed1=embedding_g0, embed2=embeddings_x0).contiguous()
        x1 = torch.cat([x_g0, x1], dim=1)
        # SFA
        # print('=================', x1.size(), embedding_g0.size())
        x1 = self.sa1_1(x1, x1).contiguous().float()  # , embed1=embedding_g0, embed2=embedding_g0

        # GDP
        num_needed = int(N // 4)  # 16 times  * self.ratio
        idx_1 = flattened.new_zeros((batch_size, num_needed), dtype=torch.int)
        for i in range(batch_size):
            idx_1[i] = torch.randperm(N // 2, device=points.device)[:num_needed]

        # idx_1 = pn2_utils.furthest_point_sample(points.permute(0, 2, 1).contiguous(), N // 4)
        x_g1 = pn2_utils.gather_operation(x1.contiguous(), idx_1)  # B,C,N
        points = pn2_utils.gather_operation(points, idx_1)
        # xyz = points.permute(0, 2, 1).contiguous()
        # embeddings_2 = embed(xyz, self.basis)
        # embeddings_2 = self.embed_2(torch.cat([xyz, embeddings_2], dim=2))  # 3 + 48 -> 256
        # x_g1 = pn2_utils.gather_operation(x1, idx_1)
        # points = pn2_utils.gather_operation(points, idx_1)

        # embedding_g1 = pn2_utils.gather_operation(embedding_g0.float(), x_g1)
        x2 = self.sa2(x_g1, x1).contiguous()  # C*2, N , embed1=embedding_g1, embed2=embedding_g0
        x2 = torch.cat([x_g1, x2], dim=1)
        # SFA
        x2 = self.sa2_1(x2, x2).contiguous().float()  # , embed1=embedding_g1, embed2=embedding_g1

        # # GDP
        # # idx_2 = pn2_utils.furthest_point_sample(points.permute(0, 2, 1).contiguous(), N // 8)
        # num_needed = int(N // 8)  # 16 times  * self.ratio
        # idx_2 = flattened.new_zeros((batch_size, num_needed), dtype=torch.int)
        # for i in range(batch_size):
        #     idx_2[i] = torch.randperm(N // 4, device=points.device)[:num_needed]
        # x_g2 = pn2_utils.gather_operation(x2, idx_2)  # B,C,N
        # # x_g2 = pn2_utils.gather_operation(x2, idx_2)
        # points = pn2_utils.gather_operation(points, idx_2)
        # xyz = points.permute(0, 2, 1).contiguous()
        # embeddings_3 = embed(xyz, self.basis)
        # embeddings_3 = self.embed_3(torch.cat([xyz, embeddings_3], dim=2))  # 3 + 48 -> 256
        # x3 = self.sa3(x_g2, x2, embed1=embeddings_3, embed2=embeddings_2).contiguous()  # C*4, N/4
        # x3 = torch.cat([x_g2, x3], dim=1)
        # # SFA
        # x3 = self.sa3_1(x3, x3, embed1=embeddings_3, embed2=embeddings_3).contiguous()

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
        sparse_feature = self.conv_out(x2)
        # print('=================output feature size: ', sparse_feature.size())  # torch.Size([B, C=256, N=128])
        sparse_feature = sparse_feature.permute(0, 2, 1).contiguous()
        points = points.permute(0, 2, 1).contiguous()
        return sparse_feature, points
