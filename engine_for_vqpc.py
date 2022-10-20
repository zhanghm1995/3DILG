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


def train_batch(model, gt_pc, radius, criterion):
    outputs, _, _, loss_vq, _ = model(gt_pc)

    emd_loss = 100.0 * compute_emd_loss(outputs, gt_pc, radius)
    # rep_loss = 10.0 * compute_repulsion_loss(outputs)
    uniform_loss = 10.0 * get_uniform_loss(outputs)

    loss = loss_vq + emd_loss + uniform_loss

    return loss, outputs, emd_loss.item(), uniform_loss.item(), loss_vq.item()


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
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

    for data_iter_step, (input_data, gt_data, radius_data) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
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

        gt_data = gt_data.to(device, non_blocking=True)
        radius_data = radius_data.to(device, non_blocking=True)

        if loss_scaler is None:
            raise NotImplementedError
        else:
            with torch.cuda.amp.autocast(enabled=True):

                loss, outputs, emd_loss, uniform_loss, loss_vq = \
                    train_batch(model, gt_data, radius_data, criterion)
        
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
        metric_logger.update(loss_emd=emd_loss)
        # metric_logger.update(loss_rep=rep_loss)
        metric_logger.update(uniform_loss=uniform_loss)
        metric_logger.update(loss_vq=loss_vq)


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
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.update(loss_emd=emd_loss, head="opt")
            # log_writer.update(loss_rep=rep_loss, head="opt")
            log_writer.update(loss_uni=uniform_loss, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(epoch, log_dir, data_loader, model, device, visualize=True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    batch_index = -1
    for batch in metric_logger.log_every(data_loader, 200, header):
        batch_index += 1

        _, gt_data, radius_data = batch

        gt_pc = gt_data.float().to(device, non_blocking=True).contiguous()
        radius = radius_data.float().to(device, non_blocking=True).contiguous()

        with torch.cuda.amp.autocast():
            pred, _, _, loss_vq, _ = model(gt_pc)
            pred = pred.float().cuda().contiguous()
            cd_loss = compute_cd_loss(pred, gt_pc)
            emd_loss = compute_emd_loss(pred, gt_pc, radius)
            uniform_loss = get_uniform_loss(pred)
        
        batch_size = gt_pc.shape[0]
        metric_logger.update(emd_loss=emd_loss.item())
        metric_logger.update(cd_loss=cd_loss.item())
        metric_logger.update(vq_loss=loss_vq.item())
        metric_logger.update(uniform_loss=uniform_loss.item())


        ## Save point cloud for visualization
        if visualize and batch_index % 100 == 0:
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
        pred[output>=0] = 1

        accuracy = (pred==labels).float().sum(dim=1) / labels.shape[1]
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