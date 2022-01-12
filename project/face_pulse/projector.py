"""Face Gan3 Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import image_likeness
import todos

import pdb

def best_wscode(
        G,
        target,  # 1xCxHxW, [0,1.0], W & H must match G output resolution
        num_steps=1000):

    device = G.device
    G = copy.deepcopy(G).eval().requires_grad_(False)

    # Compute w stats.
    w_avg_samples = 10
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    # w_samples = w_samples[:, :1, :]  # [N, 1, C]

    w_avg = torch.mean(w_samples, dim=0, keepdim=True)  # [1, L, C]
    w_std = (torch.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    del w_samples
    torch.cuda.empty_cache()

    target = target.to(device)
    # Features for target image. Reshape to 256x256 if it's larger to use with VGG16
    if target.shape[2] > 256:
        # target = F.interpolate(target, size=(256, 256), mode='area')
        target = F.interpolate(target, size=(256, 256))

    target_height, target_width = target.shape[2], target.shape[3]
    todos.data.save_tensor(target, "/tmp/dell_resize.png")

    # Load the feature detector.
    loss_model = image_likeness.get_model("vgg16")
    loss_model = loss_model.eval().to(device)
    # loss_model = copy.deepcopy(loss_model).requires_grad_(False)

    # mse = nn.MSELoss(reduction='mean')

    w_opt = w_avg.clone().detach().requires_grad_(True)
    initial_learning_rate = 0.5
    optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)

    min_loss = 1e10;
    best_ws = w_opt.detach()

    pbar = tqdm(total=num_steps)
    for step in range(num_steps):
        t = step / num_steps

        initial_noise_factor = 0.05
        noise_ramp_length = 0.75
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2

        # Learning rate schedule.
        lr_rampdown_length = 0.25
        lr_rampup_length = 0.05
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = w_opt + w_noise
        # ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        # Resize synth_images for likeness
        if synth_images.shape[2] != target_height or synth_images.shape[3] != target_width:
            # synth_images = F.interpolate(synth_images, size=(target_height, target_width), mode='area')
            synth_images = F.interpolate(synth_images, size=(target_height, target_width))

        if step % 10 == 0:
            todos.data.save_tensor(synth_images, f"output/dell_{step:06d}.png")

        # loss = mse(synth_images, target) + 10.0 * loss_model(synth_images, target)
        loss = loss_model(synth_images, target)

        pbar.set_postfix(loss="{:.6f}".format(loss.item()))
        pbar.update(1)

        # Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del synth_images
        torch.cuda.empty_cache()

        # Save best ws
        if loss.item() < min_loss:
            min_loss = loss.item()
            best_ws = ws.clone().detach()

    del ws
    torch.cuda.empty_cache()

    return best_ws

