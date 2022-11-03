'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-11-01 14:23:50
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-10-13 10:39:44
Email: haimingzhang@link.cuhk.edu.cn
Description: The evaluation code of Pytorch version.
'''

import os
import os.path as osp
import torch
import numpy as np
from glob import glob


def normalize_point_cloud(points):
    """Normalize point cloud to have zero mean and unit variance.

    Args:
        points (_type_): (B, N, 3)

    Returns:
        _type_: _description_
    """
    if isinstance(points, torch.Tensor):
        points_xyz = points[..., :3]
        centroid = torch.mean(points_xyz, axis=1, keepdims=True)
        furthest_distance = torch.max(
            torch.sqrt(torch.sum((points_xyz - centroid) ** 2, axis=-1)), dim=1, keepdims=True)[0]

        point_cloud_out = points.clone()
        point_cloud_out[..., :3] -= centroid
        point_cloud_out[..., :3] /= furthest_distance[..., None]
    else:
        centroid = np.mean(points[..., :3], axis=1, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((points[..., :3] - centroid) ** 2, axis=-1)), axis=1, keepdims=True)

        point_cloud_out = points.copy()
        point_cloud_out[..., :3] -= centroid
        point_cloud_out[..., :3] /= np.expand_dims(furthest_distance, axis=-1)
    return point_cloud_out


# def compute_cd_dist(pred, gt):
#     """Compute the Chamfer Distance borrowed from https://github.com/otaheri/chamfer_distance,
#     you need to install the chamfer_distance package by using:

#     pip install git+'https://github.com/otaheri/chamfer_distance'

#     Args:
#         pred (tensor): (B, N, 3)
#         gt (tensor): (B, N, 3)

#     Returns:
#         tensor: scalar tensor
#     """
#     from chamfer_distance import ChamferDistance as chamfer_dist

#     dist1, dist2, idx1, idx2 = chamfer_dist(pred, gt)
#     dist = (torch.mean(dist1)) + (torch.mean(dist2))
#     return dist


def compute_cd_dist(pred, gt):
    """Compute the Chamfer Distance borrowed from https://github.com/otaheri/chamfer_distance,
    you need to install the chamfer_distance package by using:

    pip install git+'https://github.com/otaheri/chamfer_distance'

    Args:
        pred (tensor): (B, N, 3)
        gt (tensor): (B, N, 3)

    Returns:
        tensor: scalar tensor
    """
    from chamfer_distance import ChamferDistance as chamfer_dist

    dist1, dist2, idx1, idx2 = chamfer_dist(pred, gt)
    dist_mean = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    dist = (torch.mean(dist1)) + (torch.mean(dist2))
    bad_idx = torch.argmax(dist_mean).cpu().item()
    print(dist_mean, bad_idx)

    return dist, bad_idx


def compute_hd_distance(pred, gt):
    from chamfer_distance import ChamferDistance as chamfer_dist

    dist1, dist2, idx1, idx2 = chamfer_dist(pred, gt)
    max_dist1 = torch.max(dist1, dim=-1, keepdim=True)[0]
    max_dist2 = torch.max(dist2, dim=-1, keepdim=True)[0]
    
    hd_value = torch.max(max_dist1 + max_dist2, dim=-1)[0]
    hd_value = torch.mean(hd_value)
    return hd_value


def compute_cd_hd_distance(pred, gt, lamda_cd=1, lamda_hd=1):
    from chamfer_distance import ChamferDistance as chamfer_dist

    dist1, dist2, idx1, idx2 = chamfer_dist(pred, gt)
    
    cd_dist = (torch.mean(dist1)) + (torch.mean(dist2))

    max_dist1 = torch.max(dist1, dim=-1, keepdim=True)[0]
    max_dist2 = torch.max(dist2, dim=-1, keepdim=True)[0]
    
    hd_value = torch.max(max_dist1 + max_dist2, dim=-1)[0]
    hd_value = torch.mean(hd_value)
    cd_dist = lamda_cd * cd_dist
    hd_value = lamda_hd * hd_value
    return cd_dist, hd_value


def evaluate_online(pred_dir, gt_dir, device):
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".xyz")])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".xyz")])

    pred_fp_list, gt_fp_list = [], []
    for pred_file_name in pred_files:
        gt_index = gt_files.index(pred_file_name)
        gt_file_name = gt_files[gt_index]

        pred_fp_list.append(osp.join(pred_dir, pred_file_name))
        gt_fp_list.append(osp.join(gt_dir, gt_file_name))

    print(f"The length of pred_pc_list: {len(pred_fp_list)}, gt_pc_list: {len(gt_fp_list)}")
    print(pred_fp_list)
    
    pred_pc_list, gt_pc_list = [], []
    for pred_fp, gt_fp in zip(pred_fp_list, gt_fp_list):
        pred_pc = np.loadtxt(pred_fp)
        gt_pc = np.loadtxt(gt_fp)

        pred_pc_list.append(pred_pc)
        gt_pc_list.append(gt_pc)
    
    pred_pc_arr = torch.from_numpy(np.stack(pred_pc_list, axis=0)).to(torch.float32).to(device)
    gt_pc_arr = torch.from_numpy(np.stack(gt_pc_list, axis=0)).to(torch.float32).to(device)

    ## Normalize the point cloud
    pred = normalize_point_cloud(pred_pc_arr)
    gt = normalize_point_cloud(gt_pc_arr)

    # ## Compute the chamfer distance    
    cd_dist, bad_idx = compute_cd_dist(pred, gt)
    print("Mean CD dist: {:.8f}".format(cd_dist))
    hd_dist = compute_hd_distance(pred, gt)
    print("Mean HD dist: {:.8f}".format(hd_dist))
    
    print(f"The bad one is: ", pred_fp_list[bad_idx])


if __name__ == "__main__":
    pred_dir = "/mntnfs/cui_data4/yanchengwang/3DILG/visualize/stage1/"
    gt_dir = "/mntnfs/cui_data4/yanchengwang/PUGAN-pytorch-master/data/test/gt_FPS_8192/"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    evaluate_online(pred_dir, gt_dir, device)
