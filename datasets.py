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
        return PUGAN_Dataset(args.data_path, split=None, isTrain=True)
    elif split == 'val':
        # return PUGAN_Dataset(args.data_path, split="./data/train/val.txt", isTrain=False)
        return xyz_Dataset_Whole(data_dir='./data/test/gt_FPS_8192/', n_input=2048)
    elif split == 'PUGAN_input_2048':
        return xyz_Dataset_Whole(data_dir='./data/test/gt_FPS_8192/', n_input=2048)
    elif split == 'PU1K_input_2048':
        return xyz_Pair_Dataset(lr_dir='./data/PU1K/test/input_2048/input_2048/', gt_dir='./data/PU1K/test/input_2048/gt_8192/', n_input=2048)
    elif split == 'PU1K_input_1024':
        return xyz_Pair_Dataset(lr_dir="./data/PU1K/test/input_1024/input_1024/", gt_dir="./data/PU1K/test/input_1024/gt_4096/", n_input=1024)
    elif split == 'PU1K_input_512':
        return xyz_Pair_Dataset(lr_dir="./data/PU1K/test/input_512/input_512/", gt_dir="./data/PU1K/test/input_512/gt_2048/", n_input=512)
    elif split == 'PU1K_input_256':
        return xyz_Pair_Dataset(lr_dir="./data/PU1K/test/input_256/input_256/", gt_dir="./data/PU1K/test/input_256/gt_1024/", n_input=256)
    elif split == 'PUGAN_input_1024':
        return xyz_Pair_Dataset(lr_dir="./data/CVPR_data/input_1024/", gt_dir="./data/CVPR_data/gt_FPS_4096/", n_input=1024)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    pass