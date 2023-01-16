'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-10-20 19:46:18
Email: haimingzhang@link.cuhk.edu.cn
Description: The PU losses
'''

import torch

from chamfer_distance import chamfer_distance as chamfer_dist
from knn_cuda import KNN
from auction_match import auction_match
from pointnet2.utils import pointnet2_utils as pn2_utils

from pointnet2.utils.utils import knn_point
import math

def compute_emd_loss(pred, gt, pcd_radius):
    idx, _ = auction_match(pred, gt)
    matched_out = pn2_utils.gather_operation(gt.transpose(1, 2).contiguous(), idx)
    matched_out = matched_out.transpose(1, 2).contiguous()
    dist2 = (pred - matched_out) ** 2
    dist2 = dist2.view(dist2.shape[0], -1)
    dist2 = torch.mean(dist2, dim=1, keepdims=True)
    dist2 /= pcd_radius
    return torch.mean(dist2)


def compute_cd_loss(pred, gt):
    dist1, dist2 = chamfer_dist(gt, pred)
    cost = (torch.mean(dist1)) + (torch.mean(dist2))
    return cost


def compute_repulsion_loss(pred, nn_size=5, radius=0.07, h=0.03, eps=1e-12):
    _, idx = knn_point(nn_size, pred, pred, transpose_mode=True)
    idx = idx[:, :, 1:].to(torch.int32)  # remove first one
    idx = idx.contiguous()  # B, N, nn

    pred = pred.transpose(1, 2).contiguous()  # B, 3, N
    grouped_points = pn2_utils.grouping_operation(pred, idx)  # (B, 3, N), (B, N, nn) => (B, 3, N, nn)

    grouped_points = grouped_points - pred.unsqueeze(-1)
    dist2 = torch.sum(grouped_points ** 2, dim=1)
    dist2 = torch.max(dist2, torch.tensor(eps).cuda())
    dist = torch.sqrt(dist2)
    weight = torch.exp(- dist2 / h ** 2)

    uniform_loss = torch.mean((radius - dist) * weight)
    return uniform_loss

def get_uniform_loss(pcd, percentage=[0.004, 0.006, 0.008, 0.010, 0.012], radius=1.0):
    B, N, C = pcd.shape[0], pcd.shape[1], pcd.shape[2]
    npoint = int(N * 0.05)
    loss = 0
    further_point_idx = pn2_utils.furthest_point_sample(pcd.permute(0, 2, 1).contiguous(), npoint)
    new_xyz = pn2_utils.gather_operation(pcd.permute(0, 2, 1).contiguous(), further_point_idx)  # B,C,N
    for p in percentage:
        nsample = int(N * p)
        r = math.sqrt(p * radius)
        disk_area = math.pi * (radius ** 2) / N

        idx = pn2_utils.ball_query(r, nsample, pcd.contiguous(),
                                   new_xyz.permute(0, 2, 1).contiguous())  # b N nsample

        expect_len = math.sqrt(disk_area)

        grouped_pcd = pn2_utils.grouping_operation(pcd.permute(0, 2, 1).contiguous(), idx)  # B C N nsample
        grouped_pcd = grouped_pcd.permute(0, 2, 3, 1)  # B N nsample C

        grouped_pcd = torch.cat(torch.unbind(grouped_pcd, dim=1), dim=0)  # B*N nsample C
        knn_uniform = KNN(k=2, transpose_mode=True)
        dist, _ = knn_uniform(grouped_pcd, grouped_pcd)
        # print(dist.shape)
        uniform_dist = dist[:, :, 1:]  # B*N nsample 1
        uniform_dist = torch.abs(uniform_dist + 1e-8)
        uniform_dist = torch.mean(uniform_dist, dim=1)
        uniform_dist = (uniform_dist - expect_len) ** 2 / (expect_len + 1e-8)
        mean_loss = torch.mean(uniform_dist)
        mean_loss = mean_loss * math.pow(p * 100, 2)
        loss += mean_loss
    return loss / len(percentage)
