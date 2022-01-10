"""Face Pulse Package."""  # coding=utf-8
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
import numpy as np

import pdb


def get_model():
    # checkpoint = os.path.dirname(__file__) + "/models/image_stylegan3.pth"
    # model = stylegan3.Generator()
    checkpoint = "/tmp/image_stylegan3.pth"
    model = stylegan3.Generator(img_resolution=256)

    todos.model.load(model, checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    # Save device as model attribute
    model.device = device

    return model


def model_forward(model, input_tensor):
    input_tensor = input_tensor.to(model.device)
    label = None
    # torch.zeros([1, model.c_dim]).to(model.device)
    with torch.no_grad():
        output_tensor = model(input_tensor, label, truncation_psi=0.70, noise_mode="const")
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
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        Image.fromarray(img[0].cpu().numpy(), "RGB").save(f"{output_dir}/seed_{seed:06d}.png")
    print("Total spend time:", time.time() - start_time)

