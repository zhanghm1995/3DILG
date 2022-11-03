'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-09-29 11:42:32
Email: haimingzhang@link.cuhk.edu.cn
Description: The PU-Net dataloader, borrowed from https://github.com/UncleMEDM/PUGAN-pytorch
'''

import torch
import h5py
import torch.utils.data as data
import os, sys

import numpy as np
import data_util as utils
from torchvision import transforms
from torch.utils.data._utils.collate import default_collate
from easydict import EasyDict

import point_operation


def load_h5_data(h5_filename='', opts=None, skip_rate=1, use_randominput=True):
    num_point = opts.num_point
    num_4X_point = int(opts.num_point * 4)
    num_out_point = int(opts.num_point * opts.up_ratio)

    print("h5_filename : ", h5_filename)
    if use_randominput:
        print("use randominput, input h5 file is:", h5_filename)
        f = h5py.File(h5_filename)
        input = f['poisson_%d' % num_4X_point][:]
        gt = f['poisson_%d' % num_out_point][:]
    else:
        print("Do not randominput, input h5 file is:", h5_filename)
        f = h5py.File(h5_filename)
        input = f['poisson_%d' % num_point][:]
        gt = f['poisson_%d' % num_out_point][:]

    # name = f['name'][:]
    assert len(input) == len(gt)

    print("Normalization the data")
    data_radius = np.ones(shape=(len(input)))
    centroid = np.mean(gt[:, :, 0:3], axis=1, keepdims=True)
    gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(gt[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
    gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
    input[:, :, 0:3] = input[:, :, 0:3] - centroid
    input[:, :, 0:3] = input[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)

    input = input[::skip_rate]
    gt = gt[::skip_rate]
    data_radius = data_radius[::skip_rate]
    print("total %d samples" % (len(input)))
    return input, gt, data_radius


class PUGANDataset(data.Dataset):
    def __init__(self, h5_fp=None, num_point=256, up_ratio=4, patch_num_point=256,
                 use_random_input=True, augment=True, jitter_sigma=0.01, jitter_max=0.03):
        super().__init__()

        self.augment = augment
        self.use_random_input = use_random_input
        self.patch_num_point = patch_num_point
        self.jitter_sigma = jitter_sigma
        self.jitter_max = jitter_max

        self.opts = EasyDict(num_point=num_point, up_ratio=up_ratio)
        self.input_data, self.gt_data, self.radius_data = load_h5_data(
            h5_fp, opts=self.opts, use_randominput=use_random_input)

    def __getitem__(self, index):
        input_data = self.input_data[index]
        gt_data = self.gt_data[index]
        radius_data = self.radius_data[index]
        return input_data, gt_data, radius_data

    def __len__(self):
        return len(self.input_data)

    def custom_collate_fn(self, batch):
        res_batch = default_collate(batch)

        batch_input_data, batch_data_gt, radius = [entry.numpy().copy() for entry in res_batch]

        batch_size = batch_input_data.shape[0]

        if self.use_random_input:
            new_batch_input = np.zeros((batch_size, self.patch_num_point, batch_input_data.shape[2]))
            for i in range(batch_size):
                idx = point_operation.nonuniform_sampling(batch_input_data.shape[1], sample_num=self.patch_num_point)
                new_batch_input[i, ...] = batch_input_data[i][idx]
            batch_input_data = new_batch_input

        if self.augment:
            batch_input_data = point_operation.jitter_perturbation_point_cloud(
                batch_input_data, sigma=self.jitter_sigma, clip=self.jitter_max)
            batch_input_data, batch_data_gt = point_operation.rotate_point_cloud_and_gt(batch_input_data, batch_data_gt)
            batch_input_data, batch_data_gt, scales = \
                point_operation.random_scale_point_cloud_and_gt(batch_input_data,
                                                                batch_data_gt,
                                                                scale_low=0.8, scale_high=1.2)
            radius = radius * scales

        return (torch.FloatTensor(batch_input_data), torch.FloatTensor(batch_data_gt), torch.FloatTensor(radius))


class xyz_Dataset_Whole(data.Dataset):
    def __init__(self, data_dir, n_input=2048, need_norm=True):
        
        super().__init__()
        
        self.raw_input_points = 8192
        self.n_input = n_input
        self.need_norm = need_norm

        file_list = [f for f in os.listdir(data_dir) if f.endswith(".xyz")]
        self.names = [x.split('.')[0] for x in file_list]
        self.sample_path = [os.path.join(data_dir, x) for x in file_list]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        gt_points = np.loadtxt(self.sample_path[index]) # (N, 3)
        
        ## Build the LR input
        choice = np.random.choice(gt_points.shape[0], self.n_input, replace=True)
        lr_points = gt_points[choice]

        if self.need_norm:
            centroid = np.mean(gt_points[:, :3], axis=0)
            dist = np.linalg.norm(gt_points[:, :3] - centroid, axis=1)
            furthest_dist = np.max(dist)

            gt_points = (gt_points - centroid) / furthest_dist
            lr_points = gt_points[choice]
            return gt_points, lr_points, furthest_dist, centroid, name
        return gt_points, lr_points, name


class PUNET_Dataset(data.Dataset):
    def __init__(self, h5_file_path='../Patches_noHole_and_collected.h5', split_dir='./train_list.txt',
                 skip_rate=1, npoint=1024, use_random=True, use_norm=True, isTrain=True):
        super().__init__()

        self.isTrain = isTrain

        self.npoint = npoint
        self.use_random = use_random
        self.use_norm = use_norm

        h5_file = h5py.File(h5_file_path)
        self.gt = h5_file['poisson_4096'][:]  # [:] h5_obj => nparray
        self.input = h5_file['poisson_4096'][:] if use_random \
            else h5_file['montecarlo_1024'][:]
        assert len(self.input) == len(self.gt), 'invalid data'
        self.data_npoint = self.input.shape[1]

        centroid = np.mean(self.gt[..., :3], axis=1, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((self.gt[..., :3] - centroid) ** 2, axis=-1)), axis=1, keepdims=True)
        self.radius = furthest_distance[:, 0]  # not very sure?

        if use_norm:
            self.radius = np.ones(shape=(len(self.input)))
            self.gt[..., :3] -= centroid
            self.gt[..., :3] /= np.expand_dims(furthest_distance, axis=-1)
            self.input[..., :3] -= centroid
            self.input[..., :3] /= np.expand_dims(furthest_distance, axis=-1)

        self.split_dir = split_dir
        self.__load_split_file()

    def __load_split_file(self):
        index = np.loadtxt(self.split_dir)
        index = index.astype(np.int)
        self.input = self.input[index, :]
        self.gt = self.gt[index, :]
        self.radius = self.radius[index]

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, index):
        input_data = self.input[index]
        gt_data = self.gt[index]
        radius_data = np.array([self.radius[index]])

        sample_idx = utils.nonuniform_sampling(self.data_npoint, sample_num=self.npoint)
        input_data = input_data[sample_idx, :]

        if not self.isTrain:
            return input_data, gt_data, radius_data

        if self.use_norm:
            # for data aug
            input_data, gt_data = utils.rotate_point_cloud_and_gt(input_data, gt_data)
            input_data, gt_data, scale = utils.random_scale_point_cloud_and_gt(input_data, gt_data,
                                                                               scale_low=0.9, scale_high=1.1)
            input_data, gt_data = utils.shift_point_cloud_and_gt(input_data, gt_data, shift_range=0.1)
            radius_data = radius_data * scale

            # for input aug
            # if np.random.rand() > 0.5:
            #    input_data = utils.jitter_perturbation_point_cloud(input_data, sigma=0.025, clip=0.05)
            # if np.random.rand() > 0.5:
            #    input_data = utils.rotate_perturbation_point_cloud(input_data, angle_sigma=0.03, angle_clip=0.09)
        else:
            raise NotImplementedError

        return input_data, gt_data, radius_data


class PUGAN_Dataset(data.Dataset):
    def __init__(self, h5_file_path='../PUGAN_poisson_256_poisson_1024.h5', 
                 split='./train.txt',
                 isTrain=True,
                 npoint=256, 
                 use_random=True, 
                 use_norm=True, 
                 use_aug=True):
        
        super().__init__()

        self.isTrain = isTrain

        self.npoint = npoint
        self.use_random = use_random
        self.use_norm = use_norm
        self.use_aug = use_aug

        h5_file = h5py.File(h5_file_path)
        self.gt = h5_file['poisson_1024'][:]  # [:] h5_obj => nparray
        self.input = h5_file['poisson_1024'][:] if use_random \
            else h5_file['poisson_256'][:]
        assert len(self.input) == len(self.gt), 'invalid data'
        self.data_npoint = self.input.shape[1]

        centroid = np.mean(self.gt[..., :3], axis=1, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((self.gt[..., :3] - centroid) ** 2, axis=-1)), axis=1, keepdims=True)
        self.radius = furthest_distance[:, 0]  # not very sure?

        if use_norm:
            self.radius = np.ones(shape=(len(self.input)))
            self.gt[..., :3] -= centroid
            self.gt[..., :3] /= np.expand_dims(furthest_distance, axis=-1)
            self.input[..., :3] -= centroid
            self.input[..., :3] /= np.expand_dims(furthest_distance, axis=-1)
        if not isTrain:
            self._load_split_file(split)

    def _load_split_file(self, split):
        index = np.loadtxt(split).astype(np.int)
        self.input = self.input[index, :]
        self.gt = self.gt[index, :]
        self.radius = self.radius[index]

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, index):
        input_data = self.input[index]
        gt_data = self.gt[index] # (1024, 3)
        radius_data = np.array([self.radius[index]])

        sample_idx = utils.nonuniform_sampling(self.data_npoint, sample_num=self.npoint)
        input_data = input_data[sample_idx, :]

        if not self.isTrain:
            return torch.FloatTensor(input_data), torch.FloatTensor(gt_data), torch.FloatTensor(radius_data)

        if self.use_norm and self.use_aug:
            # for data aug
            input_data, gt_data = utils.rotate_point_cloud_and_gt(input_data, gt_data)
            input_data, gt_data, scale = utils.random_scale_point_cloud_and_gt(input_data, gt_data,
                                                                               scale_low=0.9, scale_high=1.1)
            input_data, gt_data = utils.shift_point_cloud_and_gt(input_data, gt_data, shift_range=0.1)
            radius_data = radius_data * scale

            # for input aug
            # if np.random.rand() > 0.5:
            #    input_data = utils.jitter_perturbation_point_cloud(input_data, sigma=0.025, clip=0.05)
            # if np.random.rand() > 0.5:
            #    input_data = utils.rotate_perturbation_point_cloud(input_data, angle_sigma=0.03, angle_clip=0.09)
        else:
            raise NotImplementedError
        return torch.FloatTensor(input_data), torch.FloatTensor(gt_data), torch.FloatTensor(radius_data)

class xyz_Dataset_Whole(data.Dataset):
    def __init__(self, data_dir='../MC_5k', n_input=2048):
        super().__init__()
        self.raw_input_points = 8192
        self.n_input = n_input

        file_list = os.listdir(data_dir)
        self.names = [x.split('.')[0] for x in file_list]
        self.sample_path = [os.path.join(data_dir, x) for x in file_list]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        random_index = np.random.choice(np.linspace(0, self.raw_input_points, self.raw_input_points, endpoint=False),
                                        self.n_input).astype(np.int)
        points = np.loadtxt(self.sample_path[index])

        centroid = np.mean(points[:, 0:3], axis=0)
        dist = np.linalg.norm(points[:, 0:3] - centroid, axis=1)
        furthest_dist = np.max(dist)
        # print('###########', furthest_dist, furthest_dist.shape)
        # radius = furthest_dist[:, 0]
        # radius = furthest_dist

        reduced_point = points[random_index][:, 0:3]

        normalized_points_LR = (reduced_point - centroid) / furthest_dist
        normalized_points_GT = (points - centroid) / furthest_dist

        return points, normalized_points_GT, normalized_points_LR, furthest_dist, centroid, name