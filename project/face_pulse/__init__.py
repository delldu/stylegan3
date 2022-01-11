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

__version__ = "1.0.0"

import os
import time
from tqdm import tqdm
import torch

import redos
import todos
from PIL import Image
from . import stylegan3
from . import projector
import numpy as np

import pdb

def anchor_latent_space(G):
    # Thanks to @RiversHaveWings and @nshepperd1
    if hasattr(G.synthesis, 'input'):
        shift = G.synthesis.input.affine(G.mapping.w_avg.unsqueeze(0))
        G.synthesis.input.affine.bias.data.add_(shift.squeeze(0))
        G.synthesis.input.affine.weight.data.zero_()

def get_model():
    model = stylegan3.Generator(img_resolution=256)
    if model.img_resolution == 256:
        checkpoint = os.path.dirname(__file__) + "/models/stylegan3_ffhqu_256.pth"
    else:
        checkpoint = os.path.dirname(__file__) + "/models/stylegan3_ffhq_1024.pth"

    todos.model.load(model, checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    anchor_latent_space(model)

    # Save device as model attribute
    model.device = device

    return model


def model_forward(model, input_tensor):
    input_tensor = input_tensor.to(model.device)
    label = None
    # torch.zeros([1, model.c_dim]).to(model.device)
    # input_tensor = np.random.RandomState(seed).randn(1, G.z_dim)
    # w = model.mapping(input_tensor.to(model.device), None)
    # w_avg = model.mapping.w_avg
    # truncation_psi = 0.7
    # w = w_avg + (w - w_avg) * truncation_psi

    with torch.no_grad():
        output_tensor = model(input_tensor, label, truncation_psi=0.7, noise_mode="const")
        # output_tensor = model.synthesis(w, noise_mode="const")

    return output_tensor


def image_client(name, input_files, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.facezoom(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")


def image_server(name, HOST="localhost", port=6379):
    # load model
    model = get_model()

    def do_service(input_file, output_file, targ):
        print(f"  clean {input_file} ...")
        try:
            input_tensor = todos.data.load_tensor(input_file)
            output_tensor = model_forward(model, input_tensor)
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except:
            return False

    return redos.image.service(name, "image_weather", do_service, HOST, port)


def image_predict(rand_seeds, output_dir="output"):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model = get_model()
    start_time = time.time()
    progress_bar = tqdm(total=len(rand_seeds))
    for seed in rand_seeds:
        progress_bar.update(1)
        input_tensor = torch.from_numpy(np.random.RandomState(seed).randn(1, model.z_dim))

        img = model_forward(model, input_tensor)
        img = (img.permute(0, 2, 3, 1) * 255.0).clamp(0, 255).to(torch.uint8)
        Image.fromarray(img[0].cpu().numpy(), "RGB").save(f"{output_dir}/seed_{seed:06d}.png")
    print("Total spend time:", time.time() - start_time)


def image_projector(input_file, output_file):
    # load model
    model = get_model()
    device = model.device

    input_tensor = todos.data.load_tensor(input_file)
    input_tensor = input_tensor.to(device)
    ws = projector.best_wscode(model, input_tensor, num_steps=100)
    with torch.no_grad():
        output_tensor = model.synthesis(ws, noise_mode='const')

    todos.data.save_tensor(output_tensor, output_file)

