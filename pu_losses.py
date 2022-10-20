'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-10-20 19:46:18
Email: haimingzhang@link.cuhk.edu.cn
Description: The PU losses
'''

import torch

from chamfer_distance import ChamferDistance as chamfer_dist
from knn_cuda import KNN

from auction_match import auction_match
from pointnet2.utils import pointnet2_utils as pn2_utils
from pointnet2.utils.utils import knn_point


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
    dist1, dist2, idx1, idx2 = chamfer_dist(gt, pred)
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
