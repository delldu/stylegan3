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


def get_model(checkpoint):
    """Create model."""

    model = stylegan3.Generator(img_resolution=1024)
    todos.model.load(model, checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    return model, device


def model_forward(model, device, input_tensor):
    input_tensor = input_tensor.to(device)
    label = None
    # torch.zeros([1, model.c_dim]).to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor, label, truncation_psi=0.75, noise_mode="const")
    return output_tensor


def image_client(name, input_files, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.weather(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")


def image_server(name, HOST="localhost", port=6379):
    # load model
    checkpoint = os.path.dirname(__file__) + "/models/image_stylegan3.pth"
    model, device = get_model(checkpoint)

    def do_service(input_file, output_file, targ):
        print(f"  clean {input_file} ...")
        try:
            input_tensor = todos.data.load_tensor(input_file)
            output_tensor = model_forward(model, device, input_tensor)
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except:
            return False

    return redos.image.service(name, "image_weather", do_service, HOST, port)


def image_predict(rand_seeds, output_dir="output"):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    checkpoint = os.path.dirname(__file__) + "/models/image_stylegan3.pth"
    # checkpoint = "/tmp/image_stylegan3.pth"
    model, device = get_model(checkpoint)

    progress_bar = tqdm(total=len(rand_seeds))
    for seed in rand_seeds:
        progress_bar.update(1)
        input_tensor = torch.from_numpy(np.random.RandomState(seed).randn(1, model.z_dim))

        img = model_forward(model, device, input_tensor)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        Image.fromarray(img[0].cpu().numpy(), "RGB").save(f"{output_dir}/seed_{seed:06d}.png")


def video_service(input_file, output_file, targ):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    checkpoint = os.path.dirname(__file__) + "/models/image_stylegan3.pth"
    model, device = get_model(checkpoint)

    print(f"  clean {input_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def clean_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        input_tensor = todos.data.frame_totensor(data)

        # convert tensor from 1x4xHxW to 1x3xHxW
        input_tensor = input_tensor[:, 0:3, :, :]
        output_tensor = model_forward(model, device, input_tensor)

        temp_output_file = "{}/{:06d}.png".format(output_dir, no)
        todos.data.save_tensor(output_tensor, temp_output_file)

    video.forward(callback=clean_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i)
        os.remove(temp_output_file)

    return True


def video_client(name, input_file, output_file):
    cmd = redos.video.Command()
    context = cmd.weather(input_file, output_file)
    redo = redos.Redos(name)
    redo.set_queue_task(context)
    print(f"Created 1 video tasks for {name}.")


def video_server(name, HOST="localhost", port=6379):
    return redos.video.service(name, "video_weather", video_service, HOST, port)


def video_predict(input_file, output_file):
    return video_service(input_file, output_file, None)
