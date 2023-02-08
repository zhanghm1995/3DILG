import numpy as np

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm

from timm.models.registry import register_model
# from timm.models.layers import to_3tuple
from timm.models.layers import drop_path, trunc_normal_

from torch.nn import Sequential as Seq
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import BatchNorm1d as BN
from torch.nn import LayerNorm as LN

from torch_cluster import fps, knn
from torch_scatter import scatter_max

from einops import rearrange
from pointnet2.utils.pointnet2_utils import furthest_point_sample
from math import log
# from dis_pu.layers import (
#     FeatureExtractor,
#     DuplicateUp,
#     CoordinateRegressor,
#     PointShuffle)
from pointnet2.utils import pointnet2_utils
from vis_util import save_xyz_file
from typing import Optional, List
from PUCRN.PUCRN_new import CRNet
from PointAttN_models.PointAttN import PCT_encoder, PCT_refine
from quantize import EMAVectorQuantizer, ori_EMAVectorQuantizer
from PUCRN.PUCRN import CRNet, ori_CRNet


def _cfg(url='', **kwargs):
    return {
    }


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
        self.embedding.weight.data.normal_(0, 1)

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
        # z = rearrange(z, 'b c h w -> b h w c').contiguous()
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
        # z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

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
            # z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward_features(self, x, pos_embed):
        # x = self.patch_embed(x)
        B, _, _ = x.size()

        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x

    def forward(self, x, pos_embed):
        x = self.forward_features(x, pos_embed)
        return x


# class Embedding(nn.Module):
#     '''
#     Not used
#     '''
#     def __init__(self, query_channel=3, latent_channel=192):
#         super(Embedding, self).__init__()
#         # self.register_buffer('B', torch.randn((128, 3)) * 2)
#
#         self.l1 = weight_norm(nn.Linear(query_channel + latent_channel, 512))
#         self.l2 = weight_norm(nn.Linear(512, 512))
#         self.l3 = weight_norm(nn.Linear(512, 512))
#         self.l4 = weight_norm(nn.Linear(512, 512 - query_channel - latent_channel))
#         self.l5 = weight_norm(nn.Linear(512, 512))
#         self.l6 = weight_norm(nn.Linear(512, 512))
#         self.l7 = weight_norm(nn.Linear(512, 512))
#         self.l_out = weight_norm(nn.Linear(512, 1))
#
#     def forward(self, x, z):
#         # x: B x N x 3
#         # z: B x N x 192
#         input = torch.cat([x, z], dim=2)
#
#         h = F.relu(self.l1(input))
#         h = F.relu(self.l2(h))
#         h = F.relu(self.l3(h))
#         h = F.relu(self.l4(h))
#         h = torch.cat((h, input), axis=2)
#         h = F.relu(self.l5(h))
#         h = F.relu(self.l6(h))
#         h = F.relu(self.l7(h))
#         h = self.l_out(h)
#         return h


def embed(input, basis):
    # print(input.shape, basis.shape)
    projections = torch.einsum(
        'bnd,de->bne', input, basis)  # .permute(2, 0, 1)
    # print(projections.max(), projections.min())
    embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
    return embeddings  # B x N x E


class Embedding(nn.Module):
    def __init__(self, query_channel=3, latent_channel=192):
        super(Embedding, self).__init__()
        # self.register_buffer('B', torch.randn((128, 3)) * 2)

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

        self.l1 = weight_norm(nn.Linear(query_channel + latent_channel + self.embedding_dim, 512))
        self.l2 = weight_norm(nn.Linear(512, 512))
        self.l3 = weight_norm(nn.Linear(512, 512))
        self.l4 = weight_norm(nn.Linear(512, 512 - query_channel - latent_channel - self.embedding_dim))
        self.l5 = weight_norm(nn.Linear(512, 512))
        self.l6 = weight_norm(nn.Linear(512, 512))
        self.l7 = weight_norm(nn.Linear(512, 512))
        self.l_out = weight_norm(nn.Linear(512, 1))

    def forward(self, x, z):
        # x: B x N x 3
        # z: B x N x 192
        # input = torch.cat([x[:, :, None].expand(-1, -1, z.shape[1], -1), z[:, None].expand(-1, x.shape[1], -1, -1)], dim=-1)
        # print(x.shape, z.shape)

        pe = embed(x, self.basis)

        input = torch.cat([x, pe, z], dim=2)

        h = F.relu(self.l1(input))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        h = torch.cat((h, input), axis=2)
        h = F.relu(self.l5(h))
        h = F.relu(self.l6(h))
        h = F.relu(self.l7(h))
        h = self.l_out(h)
        return h


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


class Decoder(nn.Module):
    def __init__(self, latent_channel=192):
        super().__init__()

        self.fc = Embedding(latent_channel=latent_channel)
        self.log_sigma = nn.Parameter(torch.FloatTensor([3.0]))
        # self.register_buffer('log_sigma', torch.Tensor([-3.0]))

        self.embed = Seq(Lin(48 + 3, latent_channel))  # , nn.GELU(), Lin(128, 128))

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

        self.transformer = VisionTransformer(embed_dim=latent_channel,
                                             depth=6,
                                             num_heads=6,
                                             mlp_ratio=4.,
                                             qkv_bias=True,
                                             qk_scale=None,
                                             drop_rate=0.,
                                             attn_drop_rate=0.,
                                             drop_path_rate=0.1,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                             init_values=0.,
                                             )

    def forward(self, latents, centers, samples):
        # kernel average
        # samples: B x N x 3
        # latents: B x T x 320
        # centers: B x T x 3

        embeddings = embed(centers, self.basis)
        embeddings = self.embed(torch.cat([centers, embeddings], dim=2))
        latents = self.transformer(latents, embeddings)

        pdist = (samples[:, :, None] - centers[:, None]).square().sum(dim=3)  # B x N x T
        sigma = torch.exp(self.log_sigma)
        weight = F.softmax(-pdist * sigma, dim=2)

        latents = torch.sum(weight[:, :, :, None] * latents[:, None, :, :], dim=2)  # B x N x 128
        preds = self.fc(samples, latents).squeeze(2)

        return preds, sigma


class PUDecoder(nn.Module):
    def __init__(self, latent_channel=192, up_ratio=4):
        super().__init__()

        self.fc = Embedding(latent_channel=latent_channel)
        self.log_sigma = nn.Parameter(torch.FloatTensor([3.0]))
        # self.register_buffer('log_sigma', torch.Tensor([-3.0]))

        self.embed = Seq(Lin(48 + 3, latent_channel))  # , nn.GELU(), Lin(128, 128))

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

        self.transformer = VisionTransformer(embed_dim=latent_channel,
                                             depth=6,
                                             num_heads=6,
                                             mlp_ratio=4.,
                                             qkv_bias=True,
                                             qk_scale=None,
                                             drop_rate=0.,
                                             attn_drop_rate=0.,
                                             drop_path_rate=0.1,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                             init_values=0.,
                                             )

        self.PCT_upsample = PointAttN_Decoder(latent_channel=latent_channel,
                                              up_ratio=up_ratio)  # CRNet(up_ratio=4)  # self.CRNet_upsample 16 times

    def forward(self, latents, centers):
        '''
        latents: FPS points quantized features
        centers: FPS points coordinates
        '''
        # kernel average
        # samples: B x N x 3
        # latents: B x T x 320
        # centers: B x T x 3
        embeddings = embed(centers, self.basis)
        embeddings = self.embed(torch.cat([centers, embeddings], dim=2))
        latents = self.transformer(latents, embeddings)  # (B, M, 256) #torch.Size([B, num_points, channel])
        latents = latents.permute(0, 2, 1).contiguous()  # torch.Size([B, channel, num_points])
        pred = self.PCT_upsample(latents, centers, )
        return pred


class PointAttN_Decoder(nn.Module):
    def __init__(self, latent_channel=192, up_ratio=4):
        super().__init__()

        self.fc = Embedding(latent_channel=latent_channel)
        self.log_sigma = nn.Parameter(torch.FloatTensor([3.0]))
        # self.register_buffer('log_sigma', torch.Tensor([-3.0]))

        self.embed = Seq(Lin(48 + 3, latent_channel))  # , nn.GELU(), Lin(128, 128))

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

        self.transformer = VisionTransformer(embed_dim=latent_channel,
                                             depth=6,
                                             num_heads=6,
                                             mlp_ratio=4.,
                                             qkv_bias=True,
                                             qk_scale=None,
                                             drop_rate=0.,
                                             attn_drop_rate=0.,
                                             drop_path_rate=0.1,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                             init_values=0.,
                                             )

        self.upsample = PCT_refine(channel=256, ratio=up_ratio)

    def forward(self, latents, centers):
        '''
        latents: FPS points quantized features [B, N, C]
        centers: FPS points coordinates [B, N, 3]
        '''
        # kernel average
        # latents: B x T x 320
        # centers: B x T x 3
        embeddings = embed(centers, self.basis)
        embeddings = self.embed(torch.cat([centers, embeddings], dim=2))  # torch.Size([B, num_points, channel])
        latents = self.transformer(latents, embeddings)  # (B, M, 256) #torch.Size([B, num_points, channel])
        latents = latents.permute(0, 2, 1).contiguous()  # torch.Size([B, channel, num_points])
        pred = self.upsample(latents, centers, embeddings.permute(0, 2, 1).contiguous())
        return pred


class Prediction_Head(nn.Module):
    def __init__(self, latent_channel=256):
        super().__init__()

        self.fc = Embedding(latent_channel=latent_channel)
        self.log_sigma = nn.Parameter(torch.FloatTensor([3.0]))
        # self.register_buffer('log_sigma', torch.Tensor([-3.0]))

        self.embed = Seq(Lin(48 + 3, latent_channel))  # , nn.GELU(), Lin(128, 128))

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

        self.transformer = VisionTransformer(embed_dim=latent_channel,
                                             depth=6,
                                             num_heads=6,
                                             mlp_ratio=4.,
                                             qkv_bias=True,
                                             qk_scale=None,
                                             drop_rate=0.,
                                             attn_drop_rate=0.,
                                             drop_path_rate=0.1,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                             init_values=0.,
                                             )
        self.K = 1024
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(latent_channel),
            nn.Linear(latent_channel, self.K, bias=False))

    def forward(self, lr_feature, centers):
        '''
        lr_feature: random points encoder features [B, N, C]
        '''
        embeddings = embed(centers, self.basis)
        embeddings = self.embed(torch.cat([centers, embeddings], dim=2))  # torch.Size([B, num_points, channel])
        lr_feature = self.transformer(lr_feature, embeddings)  # (B, M, 256) #torch.Size([B, num_points, channel])
        z_e_x = rearrange(lr_feature, 'B N C-> (B N) C').contiguous()
        logits = self.idx_pred_layer(z_e_x)  # NBC    logits: torch.Size([32*256, 1024])
        # logits = logits.permute(1, 0, 2)  # (hw)bn -> b(hw)n
        soft_one_hot = F.softmax(logits, dim=1)
        # soft_one_hot = rearrange(soft_one_hot, '(B N) C -> B C N', B=B).contiguous()
        return soft_one_hot


class PointConv(torch.nn.Module):
    def __init__(self, local_nn=None, global_nn=None):
        super(PointConv, self).__init__()
        self.local_nn = local_nn
        self.global_nn = global_nn

    def forward(self, pos, pos_dst, edge_index, basis=None):
        row, col = edge_index

        out = pos[row] - pos_dst[col]

        if basis is not None:
            embeddings = torch.einsum('bd,de->be', out, basis)
            embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=1)
            out = torch.cat([out, embeddings], dim=1)

        if self.local_nn is not None:
            out = self.local_nn(out)

        out, _ = scatter_max(out, col, dim=0, dim_size=col.max().item() + 1)

        if self.global_nn is not None:
            out = self.global_nn(out)

        return out


class Encoder(nn.Module):
    def __init__(self, N, dim=128, M=2048, num_neighbors=32, down_ratio=4):
        super().__init__()

        self.embed = Seq(Lin(48 + 3, dim))  # , nn.GELU(), Lin(128, 128))

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

        # self.conv = PointConv(local_nn=Seq(weight_norm(Lin(3+self.embedding_dim, dim))))
        self.conv = PointConv(
            local_nn=Seq(weight_norm(Lin(3 + self.embedding_dim, 256)), ReLU(True), weight_norm(Lin(256, 256))),
            global_nn=Seq(weight_norm(Lin(256, 256)), ReLU(True), weight_norm(Lin(256, dim))),
        )

        self.transformer = VisionTransformer(embed_dim=dim,
                                             depth=6,
                                             num_heads=6,
                                             mlp_ratio=4.,
                                             qkv_bias=True,
                                             qk_scale=None,
                                             drop_rate=0.,
                                             attn_drop_rate=0.,
                                             drop_path_rate=0.1,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                             init_values=0.,
                                             )

        self.M = M
        self.ratio = 1 / down_ratio  # N / M #/ 2
        self.k = num_neighbors

    def forward(self, pc, fps_sample=True, index=None):  # TODO: try fps sample
        # pc: B x N x D
        B, N, D = pc.shape
        assert N == self.M

        flattened = pc.view(B * N, D)

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened
        if index is None:
            if fps_sample:
                idx = fps(pos, batch, ratio=self.ratio)  # 0.0625
            else:
                num_needed = int(N * self.ratio)
                idx = pos.new_zeros((B, num_needed), dtype=torch.long)
                for i in range(B):
                    count = i * N
                    m_idx = torch.randperm(N, device=pc.device)[:num_needed]
                    idx[i] = count + m_idx
                idx = idx.view(-1)
        else:
            idx = index

        row, col = knn(pos, pos[idx], self.k, batch, batch[idx])
        edge_index = torch.stack([col, row], dim=0)

        x = self.conv(pos, pos[idx], edge_index, self.basis)
        pos, batch = pos[idx], batch[idx]

        x = x.view(B, -1, x.shape[-1])
        pos = pos.view(B, -1, 3)
        # print('feature size and coordinate size: ',x.size(), pos.size()) [B, 128, 256], [B, 128, 3]

        embeddings = embed(pos, self.basis)

        embeddings = self.embed(torch.cat([pos, embeddings], dim=2))  # 3 + 48 -> 256

        out = self.transformer(x, embeddings)
        if index is None:
            return out, pos, idx
        else:
            return out, pos


class Stage2PretrainEncoder(nn.Module):
    def __init__(self, N, dim=128, M=2048, num_neighbors=32, down_ratio=4):
        super().__init__()

        self.embed = Seq(Lin(48 + 3, dim))  # , nn.GELU(), Lin(128, 128))

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

        # self.conv = PointConv(local_nn=Seq(weight_norm(Lin(3+self.embedding_dim, dim))))
        self.conv = PointConv(
            local_nn=Seq(weight_norm(Lin(3 + self.embedding_dim, 256)), ReLU(True), weight_norm(Lin(256, 256))),
            global_nn=Seq(weight_norm(Lin(256, 256)), ReLU(True), weight_norm(Lin(256, dim))),
        )

        self.transformer = VisionTransformer(embed_dim=dim,
                                             depth=6,
                                             num_heads=6,
                                             mlp_ratio=4.,
                                             qkv_bias=True,
                                             qk_scale=None,
                                             drop_rate=0.,
                                             attn_drop_rate=0.,
                                             drop_path_rate=0.1,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                             init_values=0.,
                                             )

        self.M = M
        self.N = N
        self.ratio = 1 / down_ratio  # N / M #/ 2
        self.k = num_neighbors

    def forward(self, lr_pc, gt_pc):
        # pc: B x N x D
        B, N, D = lr_pc.shape  # 256
        B, M, D = gt_pc.shape  # 1024
        assert N == self.N
        assert M == self.M

        lr_flattened = lr_pc.view(B * N, D)
        gt_flattened = gt_pc.view(B * M, D)

        batch = torch.arange(B).to(gt_pc.device)  # [0,1,2,...,15]
        gt_batch = torch.repeat_interleave(batch,
                                           M)  # [0,0,0,...,1,1,1,...,15,15,15] repeat each number for N=256 times
        lr_batch = torch.repeat_interleave(batch, N)

        row, col = knn(gt_flattened, lr_flattened, self.k, gt_batch, lr_batch)
        edge_index = torch.stack([col, row], dim=0)

        x = self.conv(gt_flattened, lr_flattened, edge_index, self.basis)
        pos, batch = lr_flattened, lr_batch

        x = x.view(B, -1, x.shape[-1])
        pos = pos.view(B, -1, 3)
        # print('feature size and coordinate size: ',x.size(), pos.size()) [B, 128, 256], [B, 128, 3]

        embeddings = embed(pos, self.basis)

        embeddings = self.embed(torch.cat([pos, embeddings], dim=2))  # 3 + 48 -> 256

        out = self.transformer(x, embeddings)

        return out, pos


class Stage2Encoder(nn.Module):
    def __init__(self, N=256, dim=128, M=2048, num_neighbors=32):
        super().__init__()

        self.embed = Seq(Lin(48 + 3, dim))  # , nn.GELU(), Lin(128, 128))

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

        # self.conv = PointConv(local_nn=Seq(weight_norm(Lin(3+self.embedding_dim, dim))))
        self.conv = PointConv(
            local_nn=Seq(weight_norm(Lin(3 + self.embedding_dim, 256)), ReLU(True), weight_norm(Lin(256, 256))),
            global_nn=Seq(weight_norm(Lin(256, 256)), ReLU(True), weight_norm(Lin(256, dim))),
        )

        self.transformer = VisionTransformer(embed_dim=dim,
                                             depth=6,
                                             num_heads=6,
                                             mlp_ratio=4.,
                                             qkv_bias=True,
                                             qk_scale=None,
                                             drop_rate=0.,
                                             attn_drop_rate=0.,
                                             drop_path_rate=0.1,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                             init_values=0.,
                                             )

        self.M = M  # 1024
        self.N = N  # 256
        self.ratio = N / M  # 0.25
        self.up_ratio = M / N  # 4
        self.k = num_neighbors  # smaller 32->8
        self.step_up_rate = int(np.sqrt(self.up_ratio))

        self.first_upsample_CRNet = ori_CRNet(self.up_ratio)
        ckpt = "./pretrain/model.pth"
        weights = torch.load(ckpt)['net_state_dict']
        # pre_upsample_state_dict = {}
        # for k, v in weights.items():
        #     k = "first_upsample_CRNet." + k
        #     pre_upsample_state_dict[k] = v
        self.first_upsample_CRNet.load_state_dict(weights)
        for param in self.first_upsample_CRNet.parameters():
            param.requires_grad = False
        print('loaded pretrained CRNet and fix the weights!!!!!!')
        # self.linear_start = nn.Conv1d(3, dim, 1)

    def forward(self, pc, direct_predict_code=True, load_pretrained_upsampling_net=False, pointnet=True):
        '''
        pc: input LR point cloud: 256 points [B, 256, 3]
        N: num of points in sparse point cloud
        '''
        # TODO: upsample first
        if direct_predict_code:
            B, N, D = pc.shape
            assert N == self.N  # 256
            flattened = rearrange(pc, 'B N D -> (B N) D').contiguous()  # [B*256, 3] flattened sparse pc

            batch = torch.arange(B).to(pc.device)
            batch = torch.repeat_interleave(batch, N)

            pos = flattened
            if pointnet:
                row, col = knn(pos, pos, self.k, batch, batch)  # // 4
                edge_index = torch.stack([col, row], dim=0)
                x = self.conv(pos, pos, edge_index, self.basis)
            else:
                x = self.linear_start(pc.permute(0, 2, 1))

        else:
            if load_pretrained_upsampling_net:
                if self.training:
                    # print(pc.size()) #torch.Size([B, 256, 3])
                    [p1_pred, p2_pred, p3_pred], gt = self.first_upsample_CRNet(pc.permute(0, 2, 1))
                    # print(p3_pred.size()) #torch.Size([64, 1024, 3])
                else:
                    p3_pred = self.first_upsample_CRNet(pc.permute(0, 2, 1))
                coarse_dense_pc = p3_pred  # [B, 1024, 3]
            else:
                coarse_dense_pc = self.upsampling_stage_1(pc.permute(0, 2, 1))  # [B, 3, 512]
                coarse_dense_pc = self.upsampling_stage_2(coarse_dense_pc)  # [B, 3, 1024]
                coarse_dense_pc = coarse_dense_pc.permute(0, 2, 1)  # [B, 1024, 3]

            B, N, D = coarse_dense_pc.shape
            # import os.path as osp
            # for i in range(B):
            #     save_pred_fp = osp.join('/mntnfs/cui_data4/yanchengwang/3DILG/output/debug_preupsample/',
            #                             'pred' + str(i) + '.xyz')
            #     save_gt_fp = osp.join('/mntnfs/cui_data4/yanchengwang/3DILG/output/debug_preupsample/',
            #                           'gt' + str(i) + '.xyz')
            #     # save_file = 'visualize/{}/{}.xyz'.format('more_losses', name[i])
            #     save_xyz_file(coarse_dense_pc[i, ...], save_pred_fp)
            #     # print(coarse_dense_pc.size(), gt_pc.size())
            #     save_xyz_file(gt_pc[i, ...], save_gt_fp)
            assert N == self.M  # 1024

            # flattened = pc.view(B * N, D)
            flattened = rearrange(coarse_dense_pc, 'B N D -> (B N) D').contiguous()  # [B*1024, 3] flattened dense pc

            batch = torch.arange(B).to(pc.device)
            batch = torch.repeat_interleave(batch, N)

            pos = flattened  # torch.Size([8192, 3])
            idx = fps(pos, batch, ratio=self.ratio)  # 0.0625

            row, col = knn(pos, pos[idx], self.k, batch, batch[idx])
            edge_index = torch.stack([col, row], dim=0)

            x = self.conv(pos, pos[idx], edge_index, self.basis)
            pos, batch = pos[idx], batch[idx]

        x = x.view(B, -1, x.shape[-1])
        pos = pos.view(B, -1, 3)

        embeddings = embed(pos, self.basis)

        embeddings = self.embed(torch.cat([pos, embeddings], dim=2))

        out = self.transformer(x, embeddings)

        return out, pos  # , coarse_dense_pc # [B, 256, 3]

    def new_forward(self, pc):
        '''
        pc: input LR point cloud: 256 points [B, 256, 3]
        N: num of points in sparse point cloud
        '''
        B, N, D = pc.shape
        assert N == self.N  # 256
        flattened = rearrange(pc, 'B N D -> (B N) D').contiguous()  # [B*256, 3] flattened sparse pc

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened

        row, col = knn(pos, pos, self.k, batch, batch)  # // 4
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(pos, pos, edge_index, self.basis)

        x = x.view(B, -1, x.shape[-1])
        pos = pos.view(B, -1, 3)

        embeddings = embed(pos, self.basis)

        embeddings = self.embed(torch.cat([pos, embeddings], dim=2))

        out = self.transformer(x, embeddings)

        return out, pos  # , coarse_dense_pc # [B, 1024, 3]


class Autoencoder(nn.Module):
    def __init__(self, N, K=512, dim=256, M=2048, num_neighbors=32):
        super().__init__()

        self.encoder = Encoder(N=N, dim=dim, M=M, num_neighbors=num_neighbors)

        self.decoder = Decoder(latent_channel=dim)

        self.codebook = VectorQuantizer2(K, dim)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def encode(self, x, bins=256):
        B, _, _ = x.shape

        z_e_x, centers = self.encoder(x)  # B x T x C, B x T x 3

        centers_quantized = ((centers + 1) / 2 * (bins - 1)).long()

        z_q_x_st, loss_vq, perplexity, encodings = self.codebook(z_e_x)
        # print(z_q_x_st.shape, loss_vq.item(), perplexity.item(), encodings.shape)
        return z_e_x, z_q_x_st, centers_quantized, loss_vq, perplexity, encodings

    def forward(self, x, points):
        z_e_x, z_q_x_st, centers_quantized, loss_vq, perplexity, encodings = self.encode(x)

        centers = centers_quantized.float() / 255.0 * 2 - 1

        z_q_x = z_q_x_st

        z_q_x_st = z_q_x_st
        B, N, C = z_q_x_st.shape

        logits, sigma = self.decoder(z_q_x_st, centers, points)

        return logits, z_e_x, z_q_x, sigma, loss_vq, perplexity


class PUAutoencoder(nn.Module):
    def __init__(self, N, K=1024, dim=256, M=1024, num_neighbors=32, path=None, **kwargs):
        super().__init__()

        # self.encoder = Encoder(N=N, dim=dim, M=M, num_neighbors=num_neighbors, down_ratio=4)
        self.pretrained_encoder = Encoder(N=N, dim=dim, M=M, num_neighbors=num_neighbors, down_ratio=4)
        self.decoder = PointAttN_Decoder(latent_channel=dim, up_ratio=4)

        # self.codebook = VectorQuantizer2(K, dim)
        # self.codebook = EMAVectorQuantizer(K, dim, beta=0.25)

        self.M = M
        self.N = N
        self.patch_num_ratio = 3
        self.beta = 0.25

        # self.init_from_ckpt(path=path, fix_weights=True)

    def init_from_ckpt(self, path, fix_weights=True):
        stage1_weights = torch.load(path, map_location="cpu")["model"]
        encoder_state_dict = {}
        decoder_state_dict = {}
        for k, v in stage1_weights.items():
            if 'encoder' in k:
                k = k.split('encoder.')[-1]
                encoder_state_dict[k] = v
            elif 'decoder' in k:
                k = k.split('decoder.')[-1]
                decoder_state_dict[k] = v

        self.encoder.load_state_dict(encoder_state_dict)
        self.pretrained_encoder.load_state_dict(encoder_state_dict)  # fix
        self.decoder.load_state_dict(decoder_state_dict)  # fix

        if fix_weights:
            # for param in self.encoder.parameters():
            #     param.requires_grad = False
            for param in self.pretrained_encoder.parameters():
                param.requires_grad = False
            # for param in self.decoder.parameters():
            #     param.requires_grad = False
        print(f"Load pretrained codebook and decoder from {path}. And fixed their weights")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def encode_wo_VQ(self, x, bins=256):
        B, _, _ = x.shape
        # with torch.no_grad():
        z_e_x_GT, centers, _ = self.pretrained_encoder(x)  # B x N x C, B x N x 3
        # z_e_x_GT.requires_grad_(True)
        centers_quantized = ((centers + 1) / 2 * (bins - 1)).long()

        return z_e_x_GT, centers_quantized

    def encode(self, x, bins=256):
        B, _, _ = x.shape
        z_e_x_GT, centers, index = self.pretrained_encoder(x, index=None)  # fix weights
        # z_e_x_GT.requires_grad_()
        z_e_x, _ = self.encoder(x, index=index)  # self.encoder(x)  # B x N x C, B x N x 3

        centers_quantized = ((centers + 1) / 2 * (bins - 1)).long()
        if self.training:
            z_q_x_st, loss_vq, loss_pretrain, info = self.codebook(z_e_x, z_e_x_GT)  # B x N x C
        else:
            z_q_x_st, loss_vq, info = self.codebook(z_e_x, z_e_x_GT)
        # print('==============After VQ:', z_q_x_st.size())
        perplexity, encodings, encoding_indices = info
        # print('===================', encoding_indices.unique())
        if self.training:
            return z_e_x, z_q_x_st, centers_quantized, loss_vq, loss_pretrain, perplexity, encodings
        else:
            return z_e_x, z_q_x_st, centers_quantized, loss_vq, perplexity, encodings

    def encode_new(self, x, bins=256):
        B, _, _ = x.shape

        lr_feature, centers = self.encoder(x)  # self.encoder(x)  # B x N x C, B x N x 3

        centers_quantized = ((centers + 1) / 2 * (bins - 1)).long()
        soft_one_hot = self.predition_head(lr_feature, centers)

        if self.training:
            min_encoding_indices = torch.argmax(soft_one_hot, dim=1)
            # print(len(torch.unique(min_encoding_indices)))
            z_q = self.codebook.embedding(min_encoding_indices).view(lr_feature.shape)

            # compute loss for embedding

            loss_vq = self.beta * torch.mean((z_q.detach() - lr_feature) ** 2) + \
                      torch.mean((z_q - lr_feature.detach()) ** 2)

            # print(torch.mean(z_q**2), torch.mean(z**2))

            # preserve gradients
            z_q = lr_feature + (z_q - lr_feature).detach()

            # quant_feat = torch.einsum('btn,nc->btc', [soft_one_hot, self.codebook.embedding.weight])
            # # b(hw)c -> bc(hw) -> bchw
            # # quant_feat = quant_feat.permute(0, 2, 1).view(downsampled_feature.shape)
            # quant_feat = quant_feat.permute(0, 2, 1).view(lr_feature.shape)
            # # compute loss for embedding
            # loss_vq = self.beta * F.mse_loss(quant_feat.detach(), lr_feature)
            # # preserve gradients
            # quant_feat = lr_feature + (quant_feat - lr_feature).detach()
        else:
            _, top_idx = torch.topk(soft_one_hot, 1, dim=1)
            z_q = self.codebook.get_codebook_entry(top_idx,
                                                   shape=lr_feature.shape)  # [x.shape[0], 16, 16, 256]

        z_q = adaptive_instance_normalization(z_q, lr_feature)  # [B, N, C]

        # z_q_x_st, loss_vq, info = self.codebook(quant_feat)  # B x N x C
        # print('==============After VQ:', z_q_x_st.size())
        # perplexity, encodings, encoding_indices = info
        if self.training:
            return lr_feature, z_q, centers_quantized, loss_vq  # , perplexity, encodings
        else:
            return lr_feature, z_q, centers_quantized

    def pc_prediction(self, points):
        '''
        result: torch.Size([B, 25600, 3])
        '''
        ## get patch seed from farthestsampling
        num_points = points.shape[1]
        seed_num = int(num_points / self.M * self.patch_num_ratio)

        ## FPS sampling
        further_point_idx = pointnet2_utils.furthest_point_sample(points, seed_num)
        seed_xyz = pointnet2_utils.gather_operation(
            points.permute(0, 2, 1).contiguous(), further_point_idx)  # B,C,N
        seed_xyz = seed_xyz.permute(0, 2, 1).contiguous()

        input_list = []
        result_list = []

        top_k_neareast_idx = pointnet2_utils.knn_point(self.M, seed_xyz, points)  # (batch_size, npoint1, k)
        for i in range(top_k_neareast_idx.size()[1]):
            idx = top_k_neareast_idx[:, i, :].contiguous()  # torch.Size([1, 1024])
            patch = pointnet2_utils.gather_operation(points.permute(0, 2, 1).contiguous(),
                                                     idx)
            patch = patch.permute(0, 2, 1).contiguous()  # torch.Size([1, 1024, 3])

            ## Normalize
            centroid = torch.mean(patch, axis=1, keepdims=True)
            points_mean = patch - centroid  # (B, N, 3)
            furthest_distance = torch.max(
                torch.sqrt(torch.sum(points_mean ** 2, axis=-1)), dim=1, keepdims=True)[0]  # (B, 1)
            furthest_distance = furthest_distance[..., None]  # (B, 1, 1)

            normalized_patch = points_mean / furthest_distance

            up_point = self(normalized_patch)  # up_point: (B, N, 3) ,_
            ## UnNormalize
            up_point = up_point * furthest_distance + centroid

            input_list.append(patch)
            result_list.append(up_point)

        result = torch.concat(result_list, dim=1)
        return input_list, result

    def forward_with_VQ(self, x):
        if self.training:
            z_e_x, z_q_x_st, centers_quantized, loss_vq, loss_pretrain, perplexity, encodings = self.encode(x)
        else:
            z_e_x, z_q_x_st, centers_quantized, loss_vq, perplexity, encodings = self.encode(x)
        centers = centers_quantized.float() / 255.0 * 2 - 1
        pred = self.decoder(z_q_x_st, centers)
        if self.training:
            return pred, loss_vq, loss_pretrain
        else:
            return pred  # , loss_z, loss_vq

    def forward(self, x):
        z_e_x, centers_quantized = self.encode_wo_VQ(x)

        centers = centers_quantized.float() / 255.0 * 2 - 1

        pred = self.decoder(z_e_x, centers)

        return pred


class VQPC_stage1(nn.Module):
    def __init__(self, N, K=1024, dim=256, M=1024, num_neighbors=32, path=None, **kwargs):
        super().__init__()

        self.encoder = Encoder(N=N, dim=dim, M=M, num_neighbors=num_neighbors, down_ratio=4)
        self.pretrained_encoder = Encoder(N=N, dim=dim, M=M, num_neighbors=num_neighbors, down_ratio=4)
        self.decoder = PointAttN_Decoder(latent_channel=dim, up_ratio=4)

        # self.codebook = VectorQuantizer2(K, dim)
        self.codebook = EMAVectorQuantizer(K, dim, beta=0.25)

        self.M = M
        self.N = N
        self.patch_num_ratio = 3
        self.beta = 0.25

        self.init_from_ckpt(path=path, fix_weights=True)

    def init_from_ckpt(self, path, fix_weights=True):
        stage1_weights = torch.load(path, map_location="cpu")["model"]
        encoder_state_dict = {}
        decoder_state_dict = {}
        for k, v in stage1_weights.items():
            if 'encoder' in k:
                k = k.split('encoder.')[-1]
                encoder_state_dict[k] = v
            elif 'decoder' in k:
                k = k.split('decoder.')[-1]
                decoder_state_dict[k] = v

        self.encoder.load_state_dict(encoder_state_dict)
        self.pretrained_encoder.load_state_dict(encoder_state_dict)  # fix
        self.decoder.load_state_dict(decoder_state_dict)  # fix

        if fix_weights:
            # for param in self.encoder.parameters():
            #     param.requires_grad = False
            for param in self.pretrained_encoder.parameters():
                param.requires_grad = False
            # for param in self.decoder.parameters():
            #     param.requires_grad = False
        print(f"Load pretrained codebook and decoder from {path}. And fixed their weights")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def encode(self, x, bins=256):
        B, _, _ = x.shape
        z_e_x_GT, centers, index = self.pretrained_encoder(x, index=None)  # fix weights
        z_e_x, _ = self.encoder(x, index=index)  # B x N x C, B x N x 3

        centers_quantized = ((centers + 1) / 2 * (bins - 1)).long()
        if self.training:
            z_q_x_st, loss_vq, loss_pretrain, info = self.codebook(z_e_x, z_e_x_GT)  # B x N x C
        else:
            z_q_x_st, loss_vq, info = self.codebook(z_e_x, z_e_x_GT)
        # print('==============After VQ:', z_q_x_st.size())
        perplexity, encodings, encoding_indices = info
        # print('===================', encoding_indices.unique())
        if self.training:
            return z_e_x, z_q_x_st, centers_quantized, loss_vq, loss_pretrain  # , perplexity, encodings
        else:
            return z_e_x, z_q_x_st, centers_quantized  # , loss_vq, perplexity, encodings

    def pc_prediction(self, points):
        '''
        result: torch.Size([B, 25600, 3])
        '''
        ## get patch seed from farthestsampling
        num_points = points.shape[1]
        seed_num = int(num_points / self.M * self.patch_num_ratio)

        ## FPS sampling
        further_point_idx = pointnet2_utils.furthest_point_sample(points, seed_num)
        seed_xyz = pointnet2_utils.gather_operation(
            points.permute(0, 2, 1).contiguous(), further_point_idx)  # B,C,N
        seed_xyz = seed_xyz.permute(0, 2, 1).contiguous()

        input_list = []
        result_list = []

        top_k_neareast_idx = pointnet2_utils.knn_point(self.M, seed_xyz, points)  # (batch_size, npoint1, k)
        for i in range(top_k_neareast_idx.size()[1]):
            idx = top_k_neareast_idx[:, i, :].contiguous()  # torch.Size([1, 1024])
            patch = pointnet2_utils.gather_operation(points.permute(0, 2, 1).contiguous(),
                                                     idx)
            patch = patch.permute(0, 2, 1).contiguous()  # torch.Size([1, 1024, 3])

            ## Normalize
            centroid = torch.mean(patch, axis=1, keepdims=True)
            points_mean = patch - centroid  # (B, N, 3)
            furthest_distance = torch.max(
                torch.sqrt(torch.sum(points_mean ** 2, axis=-1)), dim=1, keepdims=True)[0]  # (B, 1)
            furthest_distance = furthest_distance[..., None]  # (B, 1, 1)

            normalized_patch = points_mean / furthest_distance

            up_point = self(normalized_patch)  # up_point: (B, N, 3) ,_
            ## UnNormalize
            up_point = up_point * furthest_distance + centroid

            input_list.append(patch)
            result_list.append(up_point)

        result = torch.concat(result_list, dim=1)
        return input_list, result

    def forward(self, x):
        if self.training:
            z_e_x, z_q_x_st, centers_quantized, loss_vq, loss_pretrain = self.encode(x)
        else:
            z_e_x, z_q_x_st, centers_quantized = self.encode(x)
        centers = centers_quantized.float() / 255.0 * 2 - 1
        pred = self.decoder(z_q_x_st, centers)
        if self.training:
            return pred, loss_vq, loss_pretrain
        else:
            return pred


def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.

    Args:
        feat (Tensor): 4D tensor. -> changed to 3D
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 3, 'The input feature should be 3D tensor.'
    b, c = size[:2]
    feat_var = feat.var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1)
    feat_mean = feat.mean(dim=2).view(b, c, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.

    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.

    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class VQPC_stage2(nn.Module):
    '''
    stage2: randomly sampled point cloud -> upsampled dense point cloud
    '''

    def __init__(self, N=256, K=1024, dim=256, M=1024, num_neighbors=32, path=None, s2_pretrain_path=None, **kwargs):
        super().__init__()
        self.Stage2Encoder = Stage2Encoder(N=N, dim=dim, M=M,
                                           num_neighbors=num_neighbors)  # sparse->feature->predict code index.

        self.pretrained_encoder = Stage2PretrainEncoder(N=N, dim=dim, M=M, num_neighbors=num_neighbors, down_ratio=4)

        self.decoder = PointAttN_Decoder(latent_channel=dim, up_ratio=4)  # PUDecoder(latent_channel=dim, up_ratio=4)

        self.codebook = EMAVectorQuantizer(K, dim, beta=0.25, update=False)  # VectorQuantizer2(K, dim)

        self.M = M
        self.N = N

        # self.n_layers = n_layers

        # self.dim_mlp = self.dim_embd * 2

        # self.position_emb = nn.Parameter(torch.zeros(dim, self.dim_embd))
        # self.feat_emb = nn.Linear(256, self.dim_embd)

        # # transformer
        # self.ft_layers = nn.Sequential(
        #     *[TransformerSALayer(embed_dim=self.dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
        #       for _ in range(self.n_layers)])

        # logits_predict head
        self.dim_embd = 256
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(self.dim_embd),
            nn.Linear(self.dim_embd, K, bias=False))

        # self.idx_pred_layer = nn.Sequential(
        #     nn.LayerNorm(self.dim_embd),
        #     nn.Linear(self.dim_embd, self.dim_embd * 2, bias=False),
        #     nn.LayerNorm(self.dim_embd * 2),
        #     nn.Linear(self.dim_embd * 2, self.dim_embd * 2, bias=False),
        #     nn.LayerNorm(self.dim_embd * 2),
        #     nn.Linear(self.dim_embd * 2, self.dim_embd, bias=False),
        #     nn.LayerNorm(self.dim_embd),
        #     nn.Linear(self.dim_embd, K, bias=False))

        self.init_from_ckpt(path=path, s2_path=s2_pretrain_path, fix_weights=True)
        self.beta = 0.25
        self.CE_loss = nn.CrossEntropyLoss(reduction='sum')

    def init_from_ckpt(self, path, s2_path, fix_weights=True):
        stage1_weights = torch.load(path, map_location="cpu")["model"]
        stage2_weights = torch.load(s2_path, map_location="cpu")["model"]
        # embedding_state_dict = {}
        s1_encoder_state_dict = {}
        s2_encoder_state_dict = {}
        decoder_state_dict = {}
        codebook_state_dict = {}
        self.codebook.embedding.weight.data = stage1_weights['codebook.embedding.weight']
        # print('+++++++++++++++++', self.codebook.embedding.weight.data)
        for k, v in stage1_weights.items():
            if 'encoder' in k:
                k = k.split('encoder.')[-1]
                s1_encoder_state_dict[k] = v
            if 'decoder' in k:
                k = k.split('decoder.')[-1]
                # if 'PUCRN' in k:
                #     k = k.replace('PUCRN', 'CRNet_upsample')
                decoder_state_dict[k] = v
            elif 'codebook' in k:
                k = k.split('codebook.')[-1]
                codebook_state_dict[k] = v

        for k, v in stage2_weights.items():
            if 'encoder' in k:
                k = k.split('encoder.')[-1]
                s2_encoder_state_dict[k] = v

        # self.codebook.load_state_dict(embedding_state_dict)
        # self.codebook.embedding.load_state_dict(stage1_weights['embedding.weight'])
        # self.codebook.embedding.weight.data = stage1_weights['params_ema']['embedding.weight']
        self.Stage2Encoder.load_state_dict(s2_encoder_state_dict, strict=False)
        self.pretrained_encoder.load_state_dict(s1_encoder_state_dict)  # fix
        self.decoder.load_state_dict(decoder_state_dict)
        self.codebook.load_state_dict(codebook_state_dict)

        if fix_weights:
            for param in self.codebook.parameters():
                param.requires_grad = False
            for param in self.pretrained_encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            for k, v in self.codebook.named_parameters():
                print(k, v.requires_grad)
        print(f"Load pretrained codebook and decoder from {path}. And fixed their weights")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def Stage2Encode(self, x, gt_pc=None, bins=256, generate_index='neareatneighbor'):
        B, _, _ = x.shape

        if self.training:
            gt_feature, centers = self.pretrained_encoder(x, gt_pc)  # fix weights
            _, _, info = self.codebook(gt_feature, None, pretrained_GT=False, stage='stage2')  # B x N x C
            perplexity, encodings, encoding_indices_GT = info
            # print(encoding_indices_GT)

        lr_feature, centers = self.Stage2Encoder(x)  # B x N x C, B x N x 3
        centers_quantized = ((centers + 1) / 2 * (bins - 1)).long()

        if generate_index == 'predict':
            # z_e_x = rearrange(fps_feature, 'B C N -> (B N) C').contiguous()  # z_e_x: torch.Size([32*256, 256])
            # print('====================', downsampled_feature.size())
            # downsampled_feature = rearrange(lr_feature, 'B N C -> B C N')
            z_e_x = rearrange(lr_feature, 'B N C-> (B N) C').contiguous()  # z_e_x: torch.Size([32*256, 256])
            # z_e_x = rearrange(downsampled_feature, 'B C N -> B N C').contiguous()  # z_e_x: torch.Size([32*256, 256])

            logits = self.idx_pred_layer(z_e_x)  # NBC    logits: torch.Size([32*256, 1024])
            # logits = logits.permute(1, 0, 2)  # (hw)bn -> b(hw)n
            soft_one_hot = F.softmax(logits, dim=1)

            if self.training:
                CE_loss = self.CE_loss(logits, encoding_indices_GT)

                # # soft_one_hot = rearrange(soft_one_hot, '(B N) C -> B N C', B=B).contiguous()
                # quant_feat = torch.einsum('bn,nc->bc', [soft_one_hot, self.codebook.embedding.weight])
                # print(soft_one_hot)
                # print('========================')
                # print(self.codebook.embedding.weight)
                # # b(hw)c -> bc(hw) -> bchw
                # quant_feat = rearrange(quant_feat, '(B N) C -> B N C', B=B)
                # print(quant_feat)
                _, top_idx = torch.topk(soft_one_hot, 1, dim=1)
                # print('=========================================================')
                # print(top_idx)
                # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                # print(encoding_indices_GT)
                # print('=========================================================')
                quant_feat = self.codebook.get_codebook_entry(top_idx, shape=lr_feature.shape)
                # self.codebook.embedding.weight[top_idx, :].view(lr_feature.shape).contiguous()
                loss_pretrain = F.mse_loss(lr_feature, gt_feature)
                # compute loss for embedding
                loss_vq = F.mse_loss(quant_feat.detach(), lr_feature)  # + \ self.beta *
                # print('=======================', CE_loss, loss_vq)
                # F.mse_loss(quant_feat, lr_feature.detach())

                # loss_pretrain = F.mse_loss(lr_feature, gt_feature)  # + F.mse_loss(quant_feat, gt_feature)

                # # preserve gradients
                quant_feat = lr_feature + (quant_feat - lr_feature).detach()

            else:
                _, top_idx = torch.topk(soft_one_hot, 1, dim=1)
                quant_feat = self.codebook.get_codebook_entry(top_idx,
                                                              shape=lr_feature.shape)  # [x.shape[0], 16, 16, 256]

            # quant_feat = adaptive_instance_normalization(quant_feat, lr_feature)  # [B, N, C]
            if self.training:
                return quant_feat, centers_quantized, CE_loss, loss_vq, loss_pretrain
            else:
                return quant_feat, centers_quantized


        elif generate_index == 'neareatneighbor':
            if self.training:
                z_q, loss_vq, loss_pretrain, info = self.codebook(lr_feature, gt_feature, stage='stage2')
                return z_q, centers_quantized, loss_vq, loss_pretrain
            else:
                z_q, loss_vq, info = self.codebook(lr_feature, stage='stage2')
                return z_q, centers_quantized

        elif generate_index == 'no_VQ':
            return lr_feature, centers_quantized
            # perplexity, encodings, encoding_indices = info

        #
        #     # z = rearrange(z, 'b c N-> b N c').contiguous()
        #     z_flattened = rearrange(lr_feature, 'B C N -> (B N) C').contiguous()
        #     # z_flattened = rearrange(fps_feature, 'B N C -> (B N) C').contiguous()
        #     # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        #
        #     d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
        #         self.embedding.weight.pow(2).sum(dim=1) - 2 * \
        #         torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight)  # 'n d -> d n'
        #     encoding_indices = torch.argmin(d, dim=1)
        #     z_q = self.embedding(encoding_indices).view(lr_feature.shape)
        #     encodings = F.one_hot(encoding_indices, self.num_tokens).type(lr_feature.dtype)
        #
        #
        #
        #     min_encoding_indices = torch.argmin(d, dim=1)
        #     quant_feat = self.codebook.embedding(min_encoding_indices).view(downsampled_feature.shape)
        #     quant_feat = downsampled_feature + (quant_feat - downsampled_feature).detach()
        #
        # centers_quantized = ((centers + 1) / 2 * (bins - 1)).long()
        #
        # # z_q_x_st, loss_vq, perplexity, encodings = self.codebook(z_e_x)
        # if self.training:
        #     return quant_feat, centers_quantized, loss_vq  # , loss_vq, perplexity, encodings
        # else:
        #     return quant_feat, centers_quantized

    def Stage2Encode_with_decoder(self, x, gt_pc=None, bins=256, generate_index='neareatneighbor'):
        B, _, _ = x.shape

        if self.training:
            gt_feature, centers = self.pretrained_encoder(x, gt_pc)  # fix weights
            # _, _, info = self.codebook(gt_feature, None, pretrained_GT=False)  # B x N x C
            # perplexity, encodings, encoding_indices_GT = info

        lr_feature, centers = self.Stage2Encoder(x, gt_pc)  # B x N x C, B x N x 3
        centers_quantized = ((centers + 1) / 2 * (bins - 1)).long()

        if generate_index == 'predict':
            z_e_x = rearrange(lr_feature, 'B N C-> (B N) C').contiguous()  # z_e_x: torch.Size([32*256, 256])
            logits = self.idx_pred_layer(z_e_x)  # NBC    logits: torch.Size([32*256, 1024])
            soft_one_hot = F.softmax(logits, dim=1)

            if self.training:
                # CE_loss = self.CE_loss(logits, encoding_indices_GT)
                quant_feat, loss_vq, loss_pretrain, _ = self.codebook(lr_feature, gt_feature, pretrained_GT=True,
                                                                      stage='stage2')

            else:
                _, top_idx = torch.topk(soft_one_hot, 1, dim=1)
                quant_feat = self.codebook.get_codebook_entry(top_idx,
                                                              shape=lr_feature.shape)  # [x.shape[0], 16, 16, 256]

            # quant_feat = adaptive_instance_normalization(quant_feat, lr_feature)  # [B, N, C]
            if self.training:
                return quant_feat, centers_quantized, loss_vq, loss_pretrain
            else:
                return quant_feat, centers_quantized
        elif generate_index == 'neareatneighbor':
            if self.training:
                z_q, loss_vq, loss_pretrain, info = self.codebook(lr_feature, gt_feature, stage='stage2')
                return z_q, centers_quantized, loss_vq, loss_pretrain
            else:
                z_q, loss_vq, info = self.codebook(lr_feature, stage='stage2')
                return z_q, centers_quantized
        elif generate_index == 'no_VQ':
            return lr_feature, centers_quantized

    def pc_prediction(self, points, stage='stage1'):
        '''
        result: torch.Size([B, 25600, 3])
        '''
        ## get patch seed from farthestsampling
        lr_num_points = points.shape[1]
        # gt_num_points = lr_num_points * 4
        # seed_num = int(gt_num_points / self.M * self.patch_num_ratio) #8192/1024*3
        seed_num = int((lr_num_points / self.N) * 3)

        ## FPS sampling
        further_point_idx = pointnet2_utils.furthest_point_sample(points, seed_num)
        seed_xyz = pointnet2_utils.gather_operation(
            points.permute(0, 2, 1).contiguous(), further_point_idx)  # B,C,N
        seed_xyz = seed_xyz.permute(0, 2, 1).contiguous()

        input_list = []
        result_list = []

        if stage == 'stage1':
            num_points = self.M
        elif stage == 'stage2':
            num_points = self.N  # 256

        top_k_neareast_idx = pointnet2_utils.knn_point(num_points, seed_xyz, points)  # (batch_size, npoint1, k)
        for i in range(top_k_neareast_idx.size()[1]):
            idx = top_k_neareast_idx[:, i, :].contiguous()  # torch.Size([1, 1024])
            patch = pointnet2_utils.gather_operation(points.permute(0, 2, 1).contiguous(),
                                                     idx)
            patch = patch.permute(0, 2, 1).contiguous()  # torch.Size([1, 1024, 3])

            ## Normalize
            centroid = torch.mean(patch, axis=1, keepdims=True)
            points_mean = patch - centroid  # (B, N, 3)
            furthest_distance = torch.max(
                torch.sqrt(torch.sum(points_mean ** 2, axis=-1)), dim=1, keepdims=True)[0]  # (B, 1)
            furthest_distance = furthest_distance[..., None]  # (B, 1, 1)

            normalized_patch = points_mean / furthest_distance

            up_point = self(normalized_patch, gt_pc=None)  # up_point: (B, N, 3)
            ## UnNormalize
            up_point = up_point * furthest_distance + centroid

            input_list.append(patch)
            result_list.append(up_point)

        result = torch.concat(result_list, dim=1)
        return input_list, result

    def forward(self, random_pc, gt_pc, only_train_classification=False):
        if only_train_classification:
            if self.training:
                z_q, centers_quantized, loss_vq, loss_pretrain = self.Stage2Encode(random_pc, gt_pc=gt_pc)

            else:
                z_q, centers_quantized = self.Stage2Encode(random_pc, gt_pc=None)

            centers = centers_quantized.float() / 255.0 * 2 - 1

            pred = self.decoder(z_q, centers)
            if self.training:
                return pred, loss_vq, loss_pretrain
            else:
                return pred

        else:
            if self.training:
                z_q, centers_quantized, loss_vq, loss_pretrain = self.Stage2Encode_with_decoder(random_pc, gt_pc=gt_pc,
                                                                                                generate_index='neareatneighbor')

            else:
                z_q, centers_quantized = self.Stage2Encode_with_decoder(random_pc, gt_pc=None,
                                                                        generate_index='neareatneighbor')

            centers = centers_quantized.float() / 255.0 * 2 - 1

            pred = self.decoder(z_q, centers)
            if self.training:
                return pred, loss_vq, loss_pretrain
            else:
                return pred

class VQPC_stage2_new_prediction_head(nn.Module):
    '''
    stage2: randomly sampled point cloud -> upsampled dense point cloud
    '''

    def __init__(self, N=256, K=1024, dim=256, M=1024, num_neighbors=32, path=None, **kwargs):
        super().__init__()
        self.Stage2Encoder = Stage2Encoder(N=N, dim=dim, M=M,
                                           num_neighbors=num_neighbors)  # sparse->feature->predict code index.
        self.pretrained_encoder = Stage2PretrainEncoder(N=N, dim=dim, M=M, num_neighbors=num_neighbors, down_ratio=4)

        self.decoder = PointAttN_Decoder(latent_channel=dim, up_ratio=4)  # PUDecoder(latent_channel=dim, up_ratio=4)

        self.codebook = EMAVectorQuantizer(K, dim, beta=0.25, update=False)  # VectorQuantizer2(K, dim)

        self.M = M
        self.N = N

        # logits_predict head
        self.dim_embd = 256
        # self.idx_pred_layer = nn.Sequential(
        #     nn.LayerNorm(self.dim_embd),
        #     nn.Linear(self.dim_embd, K, bias=False))

        self.idx_pred_layer = nn.Sequential(
            nn.Conv1d(self.dim_embd, self.dim_embd, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.dim_embd),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(self.dim_embd, K, kernel_size=1),
        )
        self.init_from_ckpt(path=path, fix_weights=True)
        self.beta = 0.25
        self.CE_loss = nn.CrossEntropyLoss() #reduction='sum'

    def init_from_ckpt(self, path, fix_weights=True):
        stage_weights = torch.load(path, map_location="cpu")["model"]
        # embedding_state_dict = {}
        encoder_state_dict = {}
        decoder_state_dict = {}
        codebook_state_dict = {}
        self.codebook.embedding.weight.data = stage_weights['codebook.embedding.weight']
        # print('+++++++++++++++++', self.codebook.embedding.weight.data)
        for k, v in stage_weights.items():
            if 'encoder' in k:
                k = k.split('encoder.')[-1]
                encoder_state_dict[k] = v
            if 'decoder' in k:
                k = k.split('decoder.')[-1]
                # if 'PUCRN' in k:
                #     k = k.replace('PUCRN', 'CRNet_upsample')
                decoder_state_dict[k] = v
            elif 'codebook' in k:
                k = k.split('codebook.')[-1]
                print('========================',k,v)
                codebook_state_dict[k] = v

        # self.codebook.load_state_dict(embedding_state_dict)
        # self.codebook.embedding.load_state_dict(stage1_weights['embedding.weight'])
        # self.codebook.embedding.weight.data = stage1_weights['params_ema']['embedding.weight']
        self.pretrained_encoder.load_state_dict(encoder_state_dict)  # fix
        self.decoder.load_state_dict(decoder_state_dict)
        self.codebook.load_state_dict(codebook_state_dict)

        if fix_weights:
            for param in self.codebook.parameters():
                param.requires_grad = False
            for param in self.pretrained_encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            for k, v in self.codebook.named_parameters():
                print(k, v.requires_grad)
        print(f"Load pretrained codebook and decoder from {path}. And fixed their weights")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def Stage2Encode(self, x, gt_pc=None, bins=256, generate_index='predict'):
        B, _, _ = x.shape

        # if self.training:
        #     gt_feature, centers, pretrained_index = self.pretrained_encoder(gt_pc, index=None)  # fix weights
        #     _, _, info = self.codebook(gt_feature, None, pretrained_GT=False, stage='stage2')  # B x N x C
        #     perplexity, encodings, encoding_indices_GT = info
        #     # print(encoding_indices_GT)

        lr_feature, centers = self.Stage2Encoder(x, direct_predict_code=True, load_pretrained_upsampling_net=False, pointnet=True)
        # self.Stage2Encoder(x, direct_predict_code=False, load_pretrained_upsampling_net=True, pointnet=True)  # B x N x C, B x N x 3 direct encode the lr feature or upsample->FPS
        centers_quantized = ((centers + 1) / 2 * (bins - 1)).long()
        if self.training:
            # # map centers [B, 256, 3] to the surface of gt_pc [B, 1024, 3]
            # top_1_neareast_idx = pointnet2_utils.knn_point(1, centers, gt_pc)  # (batch_size, 256, 1)
            # for i in range(top_k_neareast_idx.size()[1]):
            #     idx = top_1_neareast_idx[:, i, :].contiguous()  # torch.Size([1, 1024])
            #     patch = pointnet2_utils.gather_operation(gt_pc.permute(0, 2, 1).contiguous(),
            #                                              idx)
            #     patch = patch.permute(0, 2, 1).contiguous()  # torch.Size([1, 1024, 3])

            gt_feature, centers = self.pretrained_encoder(x, gt_pc)  # fix weights
            _, _, info = self.codebook(gt_feature, None, pretrained_GT=False, stage='stage2')  # B x N x C
            perplexity, encodings, encoding_indices_GT = info

        if generate_index == 'predict':
            # z_e_x = rearrange(fps_feature, 'B C N -> (B N) C').contiguous()  # z_e_x: torch.Size([32*256, 256])
            # print('====================', downsampled_feature.size())
            # downsampled_feature = rearrange(lr_feature, 'B N C -> B C N')
            z_e_x = rearrange(lr_feature, 'B N C-> B C N').contiguous()  # z_e_x: torch.Size([32*256, 256])
            # z_e_x = rearrange(downsampled_feature, 'B C N -> B N C').contiguous()  # z_e_x: torch.Size([32*256, 256])

            logits = self.idx_pred_layer(z_e_x)  # NBC    logits: torch.Size([32*256, 1024])  # [B, K, N]
            # logits = logits.permute(1, 0, 2)  # (hw)bn -> b(hw)n
            soft_one_hot = F.softmax(logits, dim=1)  # [B, K, N]

            if self.training:
                logits_to_cal_CE = rearrange(logits, 'B K N-> (B N) K').contiguous()
                CE_loss = self.CE_loss(logits_to_cal_CE, encoding_indices_GT)

                # # soft_one_hot = rearrange(soft_one_hot, '(B N) C -> B N C', B=B).contiguous()
                # quant_feat = torch.einsum('bn,nc->bc', [soft_one_hot, self.codebook.embedding.weight])
                # print(soft_one_hot)
                # print('========================')
                # print(self.codebook.embedding.weight)
                # # b(hw)c -> bc(hw) -> bchw
                # quant_feat = rearrange(quant_feat, '(B N) C -> B N C', B=B)
                # print(quant_feat)
                _, top_idx = torch.topk(soft_one_hot, 1, dim=1)
                # print('=========================================================')
                # print(top_idx)
                # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                # print(encoding_indices_GT)
                # print('=========================================================')
                quant_feat = self.codebook.get_codebook_entry(top_idx, shape=lr_feature.shape)
                # self.codebook.embedding.weight[top_idx, :].view(lr_feature.shape).contiguous()
                loss_pretrain = F.mse_loss(quant_feat, gt_feature)
                # compute loss for embedding
                loss_vq = F.mse_loss(quant_feat.detach(), lr_feature)  # + \ self.beta *
                # print('=======================', CE_loss, loss_vq)
                # F.mse_loss(quant_feat, lr_feature.detach())

                # loss_pretrain = F.mse_loss(lr_feature, gt_feature)  # + F.mse_loss(quant_feat, gt_feature)

                # # preserve gradients
                quant_feat = lr_feature + (quant_feat - lr_feature).detach()

            else:
                _, top_idx = torch.topk(soft_one_hot, 1, dim=1)
                print('==========================',len(top_idx.unique()))
                quant_feat = self.codebook.get_codebook_entry(top_idx,
                                                              shape=lr_feature.shape)  # [x.shape[0], 16, 16, 256]

            # quant_feat = adaptive_instance_normalization(quant_feat, lr_feature)  # [B, N, C]
            if self.training:
                return quant_feat, centers_quantized, CE_loss, loss_vq, loss_pretrain
            else:
                return quant_feat, centers_quantized


        elif generate_index == 'neareatneighbor':
            if self.training:
                z_q, loss_vq, loss_pretrain, info = self.codebook(lr_feature, gt_feature, stage='stage2')
                return z_q, centers_quantized, loss_vq, loss_pretrain
            else:
                z_q, loss_vq, info = self.codebook(lr_feature, stage='stage2')
                return z_q, centers_quantized

        elif generate_index == 'no_VQ':
            return lr_feature, centers_quantized
            # perplexity, encodings, encoding_indices = info

    def Stage2Encode_ablation_no_VQ(self, x, bins=256):
        B, _, _ = x.shape
        lr_feature, centers = self.Stage2Encoder(x, direct_predict_code=True, load_pretrained_upsampling_net=False, pointnet=True)
        centers_quantized = ((centers + 1) / 2 * (bins - 1)).long()
        return lr_feature, centers_quantized


    def Stage2Encode_with_decoder(self, x, gt_pc=None, bins=256, generate_index='neareatneighbor'):
        B, _, _ = x.shape

        if self.training:
            gt_feature, centers = self.pretrained_encoder(x, gt_pc)  # fix weights
            # _, _, info = self.codebook(gt_feature, None, pretrained_GT=False)  # B x N x C
            # perplexity, encodings, encoding_indices_GT = info

        lr_feature, centers = self.Stage2Encoder(x, gt_pc)  # B x N x C, B x N x 3
        centers_quantized = ((centers + 1) / 2 * (bins - 1)).long()

        if generate_index == 'predict':
            z_e_x = rearrange(lr_feature, 'B N C-> (B N) C').contiguous()  # z_e_x: torch.Size([32*256, 256])
            logits = self.idx_pred_layer(z_e_x)  # NBC    logits: torch.Size([32*256, 1024])
            soft_one_hot = F.softmax(logits, dim=1)

            if self.training:
                # CE_loss = self.CE_loss(logits, encoding_indices_GT)
                quant_feat, loss_vq, loss_pretrain, _ = self.codebook(lr_feature, gt_feature, pretrained_GT=True,
                                                                      stage='stage2')

            else:
                _, top_idx = torch.topk(soft_one_hot, 1, dim=1)
                quant_feat = self.codebook.get_codebook_entry(top_idx,
                                                              shape=lr_feature.shape)  # [x.shape[0], 16, 16, 256]

            # quant_feat = adaptive_instance_normalization(quant_feat, lr_feature)  # [B, N, C]
            if self.training:
                return quant_feat, centers_quantized, loss_vq, loss_pretrain
            else:
                return quant_feat, centers_quantized
        elif generate_index == 'neareatneighbor':
            if self.training:
                z_q, loss_vq, loss_pretrain, info = self.codebook(lr_feature, gt_feature, stage='stage2')
                return z_q, centers_quantized, loss_vq, loss_pretrain
            else:
                z_q, loss_vq, info = self.codebook(lr_feature, stage='stage2')
                return z_q, centers_quantized
        elif generate_index == 'no_VQ':
            return lr_feature, centers_quantized

    def pc_prediction(self, points, stage='stage1'):
        '''
        result: torch.Size([B, 25600, 3])
        '''
        ## get patch seed from farthestsampling
        lr_num_points = points.shape[1]
        # gt_num_points = lr_num_points * 4
        # seed_num = int(gt_num_points / self.M * self.patch_num_ratio) #8192/1024*3

        seed_num = int((lr_num_points / self.N) *3) #*3

        ## FPS sampling
        further_point_idx = pointnet2_utils.furthest_point_sample(points, seed_num)
        seed_xyz = pointnet2_utils.gather_operation(
            points.permute(0, 2, 1).contiguous(), further_point_idx)  # B,C,N
        seed_xyz = seed_xyz.permute(0, 2, 1).contiguous()

        input_list = []
        result_list = []

        if stage == 'stage1':
            num_points = self.M
        elif stage == 'stage2':
            num_points = self.N  # 256

        top_k_neareast_idx = pointnet2_utils.knn_point(num_points, seed_xyz, points)  # (batch_size, npoint1, k)
        for i in range(top_k_neareast_idx.size()[1]):
            idx = top_k_neareast_idx[:, i, :].contiguous()  # torch.Size([1, 1024])
            patch = pointnet2_utils.gather_operation(points.permute(0, 2, 1).contiguous(),
                                                     idx)
            patch = patch.permute(0, 2, 1).contiguous()  # torch.Size([1, 1024, 3])

            ## Normalize
            centroid = torch.mean(patch, axis=1, keepdims=True)
            points_mean = patch - centroid  # (B, N, 3)
            furthest_distance = torch.max(
                torch.sqrt(torch.sum(points_mean ** 2, axis=-1)), dim=1, keepdims=True)[0]  # (B, 1)
            furthest_distance = furthest_distance[..., None]  # (B, 1, 1)

            normalized_patch = points_mean / furthest_distance

            up_point = self(normalized_patch, gt_pc=None)  # up_point: (B, N, 3)

            ## UnNormalize
            up_point = up_point * furthest_distance + centroid

            input_list.append(patch)
            result_list.append(up_point)

        result = torch.concat(result_list, dim=1)
        return input_list, result

    def NN_forward(self, random_pc, gt_pc, only_train_classification=True): #NN_forward
        if only_train_classification:
            # if self.training:
            #     z_q, centers_quantized, CE_loss, loss_vq, loss_pretrain = self.Stage2Encode(random_pc, gt_pc=gt_pc, generate_index='neareatneighbor')
            # else:
            #     z_q, centers_quantized = self.Stage2Encode(random_pc, gt_pc=None, generate_index='neareatneighbor')
            #
            # centers = centers_quantized.float() / 255.0 * 2 - 1
            #
            # pred = self.decoder(z_q, centers)
            # if self.training:
            #     return pred, CE_loss, loss_vq, loss_pretrain
            # else:
            #     return pred
            if self.training:
                z_q, centers_quantized, loss_vq, loss_pretrain = self.Stage2Encode(random_pc, gt_pc=gt_pc, generate_index='neareatneighbor')
            else:
                z_q, centers_quantized = self.Stage2Encode(random_pc, gt_pc=None, generate_index='neareatneighbor')

            centers = centers_quantized.float() / 255.0 * 2 - 1

            pred = self.decoder(z_q, centers)
            if self.training:
                return pred, loss_vq, loss_pretrain
            else:
                return pred

        else:
            if self.training:
                z_q, centers_quantized, loss_vq, loss_pretrain = self.Stage2Encode_with_decoder(random_pc, gt_pc=gt_pc,
                                                                                                generate_index='neareatneighbor')

            else:
                z_q, centers_quantized = self.Stage2Encode_with_decoder(random_pc, gt_pc=None,
                                                                        generate_index='neareatneighbor')

            centers = centers_quantized.float() / 255.0 * 2 - 1

            pred = self.decoder(z_q, centers)
            if self.training:
                return pred, loss_vq, loss_pretrain
            else:
                return pred

    def forward_prediction(self, random_pc, gt_pc, only_train_classification=True): # prediction forward
        if only_train_classification:
            if self.training:
                z_q, centers_quantized, CE_loss, loss_vq, loss_pretrain = self.Stage2Encode(random_pc, gt_pc=gt_pc, generate_index='predict')
            else:
                z_q, centers_quantized = self.Stage2Encode(random_pc, gt_pc=None, generate_index='predict')

            if self.training:
                return CE_loss, loss_vq, loss_pretrain
            else:
                centers = centers_quantized.float() / 255.0 * 2 - 1
                pred = self.decoder(z_q, centers)
                return pred

        else:
            if self.training:
                z_q, centers_quantized, loss_vq, loss_pretrain = self.Stage2Encode_with_decoder(random_pc,
                                                                                                gt_pc=gt_pc,
                                                                                                generate_index='neareatneighbor')

            else:
                z_q, centers_quantized = self.Stage2Encode_with_decoder(random_pc, gt_pc=None,
                                                                        generate_index='neareatneighbor')

            centers = centers_quantized.float() / 255.0 * 2 - 1

            pred = self.decoder(z_q, centers)
            if self.training:
                return pred, loss_vq, loss_pretrain
            else:
                return pred
    def forward(self, random_pc, gt_pc): #ablation no codebook forward_ablation_no_VQ
        lr_feature, centers_quantized = self.Stage2Encode_ablation_no_VQ(random_pc)
        centers = centers_quantized.float() / 255.0 * 2 - 1
        pred = self.decoder(lr_feature, centers)
        return pred

class VQPC_stage2_without_VQ(nn.Module):
    '''
    stage2: randomly sampled point cloud -> upsampled dense point cloud
    '''

    def __init__(self, N=256, dim=256, M=1024, num_neighbors=32, path=None, **kwargs):
        super().__init__()
        # self.encoder = Encoder(N=N, dim=dim, M=M, num_neighbors=num_neighbors)
        self.Stage2Encoder = Stage2Encoder(N=N, dim=dim, M=M, num_neighbors=num_neighbors)  # first upsample

        self.decoder = PUDecoder(latent_channel=dim)

        self.M = M
        self.N = N

        # self.init_from_ckpt(path=path, fix_weights=True)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def Stage2Encode(self, x, bins=256):
        B, _, _ = x.shape

        random_sample_feature, centers = self.Stage2Encoder(x)  # B x C x N, B x N x 3

        centers_quantized = ((centers + 1) / 2 * (bins - 1)).long()

        # z_q_x_st, loss_vq, perplexity, encodings = self.codebook(z_e_x)
        return random_sample_feature, centers_quantized

    def pc_prediction(self, points, stage='stage1'):
        ## get patch seed from farthestsampling
        seed1_num = 50  # int(points.size()[1] / self.M)

        ## FPS sampling
        further_point_idx = pointnet2_utils.furthest_point_sample(points, seed1_num)
        seed_xyz = pointnet2_utils.gather_operation(points.permute(0, 2, 1).contiguous(), further_point_idx)  # B,C,N
        seed_xyz = seed_xyz.permute(0, 2, 1).contiguous()

        input_list = []
        if stage == 'stage1':
            num_points = self.M
        elif stage == 'stage2':
            num_points = self.N  # 256
        top_k_neareast_idx = pointnet2_utils.knn_point(num_points, seed_xyz,
                                                       points)  # (batch_size, npoint1, k)
        for i in range(top_k_neareast_idx.size()[1]):
            idx = top_k_neareast_idx[:, i, :].contiguous()  # torch.Size([1, 1024])
            patch = pointnet2_utils.gather_operation(points.permute(0, 2, 1).contiguous(),
                                                     idx)
            patch = patch.permute(0, 2, 1).contiguous()  # torch.Size([1, 1024, 3])
            up_point = self(patch)

            input_list.append(patch)
            if i == 0:
                result = up_point
            else:
                result = torch.cat((result, up_point), dim=1)
        return input_list, result

    def forward(self, x):
        random_feat, centers_quantized = self.Stage2Encode(x)

        centers = centers_quantized.float() / 255.0 * 2 - 1

        if self.training:
            p1_pred, p2_pred, p3_pred = self.decoder(random_feat, centers)
            return p1_pred, p2_pred, p3_pred  # coarse_dense_pc.contiguous()
        else:
            p3_pred = self.decoder(random_feat, centers)
            return p3_pred


@register_model
def stage1_autoencoder(pretrained=True, **kwargs):
    model = PUAutoencoder(
        N=256,
        K=1024,  # try 2048, 4096
        M=1024,
        path=None,
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def stage1_vqpc(pretrained=True, **kwargs):
    model = VQPC_stage1(
        N=256,
        K=1024,
        M=1024,
        path="/mntnfs/cui_data4/yanchengwang/3DILG/output/stage1_autoencoder_random_4_pe_EMA_VQ_use_1024_codes_FPS_4/best_cd.pth",
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vqpc_256_2048_1024(pretrained=True, **kwargs):
    model = PUAutoencoder(
        N=256,
        K=2048,
        M=1024,
        path="/mntnfs/cui_data4/yanchengwang/3DILG/output/stage1_random_4_pe_without_VQ_offset_vq_loss_1_pretraining/best_cd.pth",
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vqpc_stage2(pretrained=False, **kwargs):
    model = VQPC_stage2(
        N=256,
        K=1024,  # 1024, 2048, 4098
        M=1024,
        path="/mntnfs/cui_data4/yanchengwang/3DILG/output/stage1_vqpc_random_4_pe_EMA_VQ_use_1024_codes_FPS_4_01_pretrain_loss/best_cd.pth",
        s2_pretrain_path="/mntnfs/cui_data4/yanchengwang/3DILG/output/stage2_FPS_4_pe_EMA_VQ_1024_vq_cd_hd_fix_decoder_upsample_first_no_VQ/best_cd.pth",
        # "/mntnfs/cui_data4/yanchengwang/3DILG/output/stage1_random_4_pe_EMA_VQ_use_pretrained_feature_as_GT_1024_codes_not_fix_resume/best_cd.pth",
        # TODO: move it to bash
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vqpc_stage2_new_prediction_head(pretrained=False, **kwargs):
    model = VQPC_stage2_new_prediction_head(
        N=256,
        K=1024,  # 1024, 2048, 4098
        M=1024,
        path="/mntnfs/cui_data4/yanchengwang/3DILG/output/stage1_vqpc_random_4_pe_EMA_VQ_use_1024_codes_FPS_4_01_pretrain_loss/best_cd.pth",
        # path = "/mntnfs/cui_data4/yanchengwang/3DILG/output/stage1_random_4_pe_EMA_VQ_use_pretrained_feature_as_GT_1024_codes_not_fix_resume/best_cd.pth",
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vqpc_stage2_without_VQ(pretrained=False, **kwargs):
    model = VQPC_stage2_without_VQ(
        N=256,
        M=1024,
        # TODO: move it to bash
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
