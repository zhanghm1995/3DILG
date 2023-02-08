import os

import torch
from torchvision import datasets, transforms

from shapenet import ShapeNet

class AxisScaling(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter
        
    def __call__(self, surface, point):
        scaling = torch.rand(1, 3) * 0.5 + 0.75
        # print(scaling)
        surface = surface * scaling
        point = point * scaling

        scale = (1 / torch.abs(surface).max().item()) * 0.999999
        surface *= scale
        point *= scale

        if self.jitter:
            surface += 0.005 * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)

        return surface, point


def build_shape_surface_occupancy_dataset(split, args):
    if split == 'train':
        transform = AxisScaling((0.75, 1.25), True)
        return ShapeNet(args.data_path, split=split, transform=transform, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
    elif split == 'val':
        return ShapeNet(args.data_path, split=split, transform=None, sampling=False, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
    else:
        return ShapeNet(args.data_path, split=split, transform=None, sampling=False, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)


def build_upsampling_dataset(split, args):
    from pu_dataset import PUGAN_Dataset, xyz_Dataset_Whole, xyz_Pair_Dataset

    if split == 'train':
        # transform = AxisScaling((0.75, 1.25), True)
        return PUGAN_Dataset(args.data_path, split=None, isTrain=True, use_random=False)
    elif split == 'val':
        # return PUGAN_Dataset(args.data_path, split="./data/train/val.txt", isTrain=False)
        return xyz_Dataset_Whole(data_dir='./data/test/gt_FPS_8192/', n_input=2048)
    elif split == 'PUGAN_input_2048':
        return xyz_Dataset_Whole(data_dir='./data/test/gt_FPS_8192/', n_input=2048)
    elif split == 'PU1K_input_2048':
        return xyz_Pair_Dataset(lr_dir='./data/PU1K/test/input_2048/input_2048/', 
                                gt_dir='./data/PU1K/test/input_2048/gt_8192/', 
                                n_input=2048)
    elif split == 'PU1K_input_1024':
        return xyz_Pair_Dataset(lr_dir="./data/PU1K/test/input_1024/input_1024/", gt_dir="./data/PU1K/test/input_1024/gt_4096/", n_input=1024)
    elif split == 'PU1K_input_512':
        return xyz_Pair_Dataset(lr_dir="./data/PU1K/test/input_512/input_512/", gt_dir="./data/PU1K/test/input_512/gt_2048/", n_input=512)
    elif split == 'PU1K_input_256':
        return xyz_Pair_Dataset(lr_dir="./data/PU1K/test/input_256/input_256/", gt_dir="./data/PU1K/test/input_256/gt_1024/", n_input=256)
    elif split == 'PUGAN_input_1024x4':
        return xyz_Pair_Dataset(lr_dir="./data/PUGAN/1024_Poisson/", gt_dir="./data/PUGAN/4096_Poisson/", n_input=1024)
    elif split == 'PUGAN_input_1024x16':
        return xyz_Pair_Dataset(lr_dir="/mntnfs/cui_data4/yanchengwang/3DILG/output/stage2_FPS_4_pe_EMA_VQ_1024_fix_decoder_pretrain_vq_match_lr_points_possion_input_ablation_NN/test_vis/epoch_090/", gt_dir="./data/PUGAN/16384_Poisson/", n_input=4096)
    elif split == 'PU1K_input_2048x16':
        return xyz_Pair_Dataset(lr_dir="/mntnfs/cui_data4/yanchengwang/3DILG/output/stage2_possion_input_ablation_no_codebook_pure_upsampling_test/PU1K_2048/epoch_000/", gt_dir="/mntnfs/cui_data4/yanchengwang/Poisson_sample/2048x16/", n_input=8192)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    pass