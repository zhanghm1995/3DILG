'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-10-20 19:43:25
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import math
import os
import os.path as osp
import sys
import numpy as np
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from timm.utils import ModelEma

import utils
from pu_losses import *
from vis_util import save_xyz_file
from pointnet2.utils import pointnet2_utils
from einops import repeat, rearrange
import numpy as np
from pu_metrics import compute_cd_hd_distance


def train_batch(model, input_pc, gt_pc, radius, stage='stage1', loss='only_vq', use_VQ=True):
    # idx = pointnet2_utils.furthest_point_sample(gt_pc, 512)
    # gt_pc_FPS_512 = pointnet2_utils.gather_operation(gt_pc.permute(0, 2, 1).contiguous(), idx)
    # gt_pc_FPS_512 = gt_pc_FPS_512.permute(0, 2, 1).contiguous()
    if stage == 'stage1':
        if use_VQ:
            loss_z, loss_zq = model(gt_pc)
        else:
            pred = model(gt_pc)
        if loss == 'only_fine_loss':
            if use_VQ:
                fine_cd_dist, fine_hd_value = compute_cd_hd_distance(pred, gt_pc, lamda_cd=100, lamda_hd=1)
                # fine_emd_loss = 100.0 * compute_emd_loss(p3_pred, gt_pc, radius)
                # loss_vq = loss_vq
                loss = loss_vq + fine_cd_dist + fine_hd_value
                return loss, loss_vq.item(), fine_cd_dist.item(), fine_hd_value.item()
            else:
                fine_cd_dist, fine_hd_value = compute_cd_hd_distance(pred, gt_pc, lamda_cd=100, lamda_hd=1)
                loss = fine_cd_dist + fine_hd_value
                return loss, fine_cd_dist.item(), fine_hd_value.item()
        elif loss == 'only_vq':
            loss = loss_z + loss_zq
            return loss, loss_z.item(), loss_zq.item()
        else:
            p1_cd_dist, p1_hd_value = compute_cd_hd_distance(p1_pred, gt_pc_FPS_512, lamda_cd=100, lamda_hd=10)
            coarse_cd_dist, coarse_hd_value = compute_cd_hd_distance(p2_pred, gt_pc, lamda_cd=100, lamda_hd=10)
            fine_cd_dist, fine_hd_value = compute_cd_hd_distance(p3_pred, gt_pc, lamda_cd=100, lamda_hd=10)

            p1_emd_loss = 100.0 * compute_emd_loss(p1_pred, gt_pc_FPS_512, radius)
            coarse_emd_loss = 100.0 * compute_emd_loss(p2_pred, gt_pc, radius)
            fine_emd_loss = 100.0 * compute_emd_loss(p3_pred, gt_pc, radius)

            # rep_loss = 10.0 * compute_repulsion_loss(outputs)
            # uniform_loss = 10.0 * get_uniform_loss(outputs)
            lamba_p1 = 0.2
            lamda_coarse = 0.5
            lamda_fine = 1

            loss = loss_vq + \
                   lamba_p1 * (p1_emd_loss + p1_cd_dist + p1_hd_value) + \
                   lamda_coarse * (coarse_emd_loss + coarse_cd_dist + coarse_hd_value) + \
                   lamda_fine * (fine_emd_loss + fine_cd_dist + fine_hd_value)

            return loss, p2_pred, p3_pred, loss_vq.item(), \
                   p1_emd_loss.item(), p1_cd_dist.item(), p1_hd_value.item(), \
                   coarse_emd_loss.item(), coarse_cd_dist.item(), coarse_hd_value.item(), \
                   fine_emd_loss.item(), fine_cd_dist.item(), fine_hd_value.item()

    elif stage == 'stage2':
        pred, loss_vq = model(input_pc)  # coarse_dense_pc
        cd_dist, hd_value = compute_cd_hd_distance(pred, gt_pc, lamda_cd=100, lamda_hd=1)

        loss = cd_dist + hd_value + loss_vq
        return loss, cd_dist.item(), hd_value.item(), loss_vq.item()

        return loss, cd_dist.item(), hd_value.item(), loss_vq.item()


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, stage=None, only_fine_loss=True, use_VQ=True):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (input_data, gt_data, radius_data) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        input_data = input_data.to(device, non_blocking=True)
        gt_data = gt_data.to(device, non_blocking=True)
        radius_data = radius_data.to(device, non_blocking=True)

        if loss_scaler is None:
            raise NotImplementedError
        else:
            with torch.cuda.amp.autocast(enabled=True):
                if stage == 'stage1':
                    if only_fine_loss:
                        if use_VQ:
                            # loss, loss_vq, fine_cd_dist, fine_hd_value = train_batch(model, gt_data, gt_data,
                            #                                                          radius_data, stage)
                            loss, loss_z, loss_zq = train_batch(model, gt_data, gt_data, radius_data, stage)
                        else:
                            loss, fine_cd_dist, fine_hd_value = train_batch(model, gt_data, gt_data, radius_data, stage)
                    else:
                        loss, p2_pred, p3_pred, loss_vq, \
                        p1_emd_loss, p1_cd_dist, p1_hd_value, \
                        coarse_emd_loss, coarse_cd_dist, coarse_hd_value, \
                        fine_emd_loss, fine_cd_dist, fine_hd_value = \
                            train_batch(model, gt_data, gt_data, radius_data, stage)
                elif stage == 'stage2':
                    loss, fine_cd_dist, fine_hd_value, loss_vq = \
                        train_batch(model, input_data, gt_data, radius_data, stage)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            raise NotImplementedError
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        if not only_fine_loss:
            metric_logger.update(loss_p1_emd=p1_emd_loss)
            metric_logger.update(loss_p1_cd=p1_cd_dist)
            metric_logger.update(loss_p1_hd=p1_hd_value)
            metric_logger.update(loss_coarse_emd=coarse_emd_loss)
            metric_logger.update(loss_coarse_cd=coarse_cd_dist)
            metric_logger.update(loss_coarse_hd=coarse_hd_value)
            metric_logger.update(loss_fine_emd=fine_emd_loss)

        # metric_logger.update(loss_fine_cd=fine_cd_dist)
        # metric_logger.update(loss_fine_hd=fine_hd_value)

        if use_VQ:
            metric_logger.update(loss_z=loss_z)
            metric_logger.update(loss_vq=loss_zq)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            if not only_fine_loss:
                log_writer.update(loss_p1_emd=p1_emd_loss, head="loss")
                log_writer.update(loss_p1_cd=p1_cd_dist, head="loss")
                log_writer.update(loss_p1_hd=p1_hd_value, head="loss")
                log_writer.update(coarse_emd_loss=coarse_emd_loss, head="loss")
                log_writer.update(coarse_cd_dist=coarse_cd_dist, head="loss")
                log_writer.update(coarse_hd_value=coarse_hd_value, head="loss")
                log_writer.update(fine_emd_loss=fine_emd_loss, head="loss")
            # log_writer.update(fine_cd_dist=fine_cd_dist, head="loss")
            # log_writer.update(fine_hd_value=fine_hd_value, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.update(loss_scale=loss_scale_value, head="opt")

            # if stage == 'stage1':
            # log_writer.update(loss_uni=uniform_loss, head="opt")
            if use_VQ:
                log_writer.update(loss_z=loss_z, head="loss")
                log_writer.update(loss_vq=loss_zq, head="loss")

            # elif stage == 'stage2':
            #     log_writer.update(pre_emd_loss=pre_emd_loss, head="loss")
            #     log_writer.update(pre_cd_dist=pre_cd_dist, head="loss")
            #     log_writer.update(pre_hd_value=pre_hd_value, head="loss")
            #     log_writer.update(loss_rep=rep_loss, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(epoch, log_dir, data_loader, model, device, visualize=True, stage='stage1', use_VQ=True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    batch_index = -1
    for batch in metric_logger.log_every(data_loader, 200, header):
        batch_index += 1

        lr_data, gt_data, radius_data = batch

        lr_pc = lr_data.float().to(device, non_blocking=True).contiguous()
        gt_pc = gt_data.float().to(device, non_blocking=True).contiguous()
        radius = radius_data.float().to(device, non_blocking=True).contiguous()

        with torch.cuda.amp.autocast():
            if stage == 'stage1':
                if use_VQ:
                    p3_pred, loss_vq = model(gt_pc)
                else:
                    p3_pred = model(gt_pc)
            elif stage == 'stage2':
                p3_pred, loss_vq = model(lr_pc)
            pred = p3_pred

            pred = pred.float().cuda().contiguous()
            cd_dist, hd_value = compute_cd_hd_distance(pred, gt_pc, lamda_cd=1, lamda_hd=1)
            emd_loss = compute_emd_loss(pred, gt_pc, radius)
            # if stage == 'stage1':
            #     uniform_loss = get_uniform_loss(pred)

        batch_size = gt_pc.shape[0]
        metric_logger.update(emd_loss=emd_loss.item())
        metric_logger.update(cd_loss=cd_dist.item())
        metric_logger.update(hd_value=hd_value.item())
        # if stage == 'stage1':
        if use_VQ:
            metric_logger.update(vq_loss=loss_vq.item())
        # metric_logger.update(uniform_loss=uniform_loss.item())

        ## Save point cloud for visualization
        if visualize and batch_index % 100 == 0 and epoch % 10 == 0:
            output_dir = osp.join(log_dir, "vis", f"epoch_{epoch:03d}")
            os.makedirs(output_dir, exist_ok=True)

            save_steps = math.floor(batch_size / 5)
            for i in range(batch_size):
                if i % save_steps != 0:
                    continue
                save_pred_fp = osp.join(output_dir, f"{batch_index:03d}_{i:03d}_pred.xyz")
                np.savetxt(save_pred_fp, pred[i, ...].cpu().numpy())

                save_gt_fp = osp.join(output_dir, f"{batch_index:03d}_{i:03d}_gt.xyz")
                np.savetxt(save_gt_fp, gt_pc[i, ...].cpu().numpy())

    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test(epoch, log_dir, data_loader, model, device, best_cd, best_hd, best_cd_epoch=0, best_hd_epoch=0,
         stage='stage1', num_GT_points=8192):
    from pu_metrics import normalize_point_cloud
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    cd_list = []
    hd_list = []
    output_dir = osp.join(log_dir, "test_vis", f"epoch_{epoch:03d}")
    if epoch % 5 == 0:
        os.makedirs(output_dir, exist_ok=True)
    for batch in metric_logger.log_every(data_loader, 200, header):
        points, normalized_points_GT, normalized_points_LR, furthest_dist, centroid, name = batch

        ori_GT = points.float().to(device, non_blocking=True).contiguous()
        normalized_points_GT = normalized_points_GT.float().to(device, non_blocking=True).contiguous()
        normalized_points_LR = normalized_points_LR.float().to(device,
                                                               non_blocking=True).contiguous()  # torch.Size([B, 2048, 3])
        furthest_dist = furthest_dist.float().to(device, non_blocking=True).contiguous()
        centroid = centroid.float().to(device, non_blocking=True).contiguous()

        with torch.cuda.amp.autocast():
            if stage == 'stage1':
                input_list, pred_pc = model.pc_prediction(normalized_points_GT)
            elif stage == 'stage2':
                input_list, pred_pc = model.pc_prediction(normalized_points_LR, stage='stage2')
            idx = pointnet2_utils.furthest_point_sample(pred_pc, num_GT_points)
            pred = pointnet2_utils.gather_operation(pred_pc.permute(0, 2, 1).contiguous(), idx)

            batch_size = points.shape[0]
            furthest_dist = repeat(furthest_dist, 'r -> r w h', w=1, h=1).contiguous()
            pred = pred * furthest_dist + centroid.unsqueeze(2).repeat(1, 1, num_GT_points)  # torch.Size([4, 3, 8192])
            pred = pred.permute(0, 2, 1).contiguous()

            pred_nomalized = normalize_point_cloud(pred)
            cd, hd_value = compute_cd_hd_distance(pred_nomalized, normalized_points_GT)
            cd_list.append(cd.data.cpu().numpy())
            hd_list.append(hd_value.data.cpu().numpy())

            pred = pred.data.cpu().numpy()
            if epoch % 5 == 0:
                for i in range(batch_size):
                    save_pred_fp = osp.join(output_dir, name[i] + '.xyz')
                    # save_file = 'visualize/{}/{}.xyz'.format('more_losses', name[i])
                    save_xyz_file(pred[i, ...], save_pred_fp)
    mean_cd = np.mean(cd_list)
    mean_hd = np.mean(hd_list)

    if mean_cd <= best_cd:
        best_cd = mean_cd
        best_cd_epoch = epoch
    if mean_hd <= best_hd:
        best_hd = mean_hd
        best_hd_epoch = epoch
    metric_logger.update(mean_cd=torch.tensor(mean_cd))
    metric_logger.update(best_cd=torch.tensor(best_cd))
    # metric_logger.update(best_cd_epoch=torch.tensor(best_cd_epoch))
    metric_logger.update(mean_hd=torch.tensor(mean_hd))
    metric_logger.update(best_hd=torch.tensor(best_hd))
    # metric_logger.update(best_hd_epoch=torch.tensor(best_hd_epoch))
    print('Epoch {}: Current cd is {}. Best cd is {} @ epoch {}.'.format(epoch, mean_cd, best_cd, best_cd_epoch))
    print('Epoch {}: Current hd is {}. Best hd is {} @ epoch {}.'.format(epoch, mean_hd, best_hd, best_hd_epoch))
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in
            metric_logger.meters.items()}, best_cd, best_hd, best_cd_epoch, best_hd_epoch


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    for batch in metric_logger.log_every(data_loader, 200, header):
        points, labels, surface, _ = batch
        surface = surface.to(device, non_blocking=True)
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            N = 100000

            output, _, _, _, _, perplexity = model(surface, points)
            loss = criterion(output, labels)

        pred = torch.zeros_like(output)
        pred[output >= 0] = 1

        accuracy = (pred == labels).float().sum(dim=1) / labels.shape[1]
        accuracy = accuracy.mean()
        intersection = (pred * labels).sum(dim=1)
        union = (pred + labels).gt(0).sum(dim=1)
        iou = intersection * 1.0 / union + 1e-5
        iou = iou.mean()

        batch_size = surface.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['iou'].update(iou.item(), n=batch_size)

    metric_logger.synchronize_between_processes()

    print('* iou{iou.global_avg:.4f} loss {losses.global_avg:.3f}'
          .format(iou=metric_logger.iou, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
