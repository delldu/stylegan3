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
import torchvision.transforms as T
import redos
import todos
from PIL import Image
from .stylegan3 import Generator
from .encoder import GradualStyleEncoder

import numpy as np

import pdb

def load_tensor(input_file):
    image = Image.open(input_file).convert("RGB").resize((256, 256))
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

def save_tensor(output_tensor, output_file):
    output_tensor = (output_tensor.permute(0, 2, 3, 1) * 255.0).clamp(0, 255).to(torch.uint8)
    Image.fromarray(output_tensor[0].cpu().numpy(), "RGB").save(output_file)


def decoder():
    """The model comes from stylegan3-t-ffhq-1024x1024.pkl."""
    cdir = os.path.dirname(__file__)
    model_path = "models/stylegan3_decoder.pth"
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = Generator(img_resolution=1024)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    # anchor_latent_space(model)

    return model

def encoder():
    """The model encoder W for stylegan3-t-ffhq-1024x1024.pkl."""
    cdir = os.path.dirname(__file__)
    model_path = "models/stylegan3_encoder.pth"
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = GradualStyleEncoder()
    # model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.load_state_dict(torch.load(checkpoint))
    model = model.eval()

    return model

# def anchor_latent_space(G):
#     # Thanks to @RiversHaveWings and @nshepperd1
#     if hasattr(G.synthesis, 'input'):
#         shift = G.synthesis.input.affine(G.mapping.w_avg.unsqueeze(0))
#         G.synthesis.input.affine.bias.data.add_(shift.squeeze(0))
#         G.synthesis.input.affine.weight.data.zero_()


def model_forward(model, device, input_tensor):
    input_tensor = input_tensor.to(device)
    label = None
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
    device = todos.model.get_device()

    # load models
    E = encoder().to(device)
    D = decoder().to(device)

    def do_service(input_file, output_file, targ):
        print(f"  face zoom {input_file} ...")
        try:
            input_tensor = load_tensor(input_file)
            with torch.no_grad():
                w = E(input_tensor)
                output_tensor = D.synthesis(w)
            save_tensor(output_tensor, output_file)
            return True
        except:
            return False

    return redos.image.service(name, "face_zoom", do_service, HOST, port)


def sample(rand_seeds, output_dir="output"):
    # Create directory to store result
    todos.data.mkdir(output_dir)
    device = todos.model.get_device()

    # load model
    D = decoder().to(device)

    start_time = time.time()
    progress_bar = tqdm(total=len(rand_seeds))
    for seed in rand_seeds:
        progress_bar.update(1)
        input_tensor = torch.from_numpy(np.random.RandomState(seed).randn(1, D.z_dim))

        output_tensor = model_forward(D, device, input_tensor)
        save_tensor(output_tensor, f"{output_dir}/seed_{seed:06d}.png")
    print("Total spend time:", time.time() - start_time)


def project(input_file, output_file):
    device = todos.model.get_device()

    # load models
    E = encoder().to(device)
    D = decoder().to(device)
    
    input_tensor = load_tensor(input_file).to(device)
    with torch.no_grad():
        w = E(input_tensor)
        output_tensor = D.synthesis(w)
    save_tensor(output_tensor, output_file)


def factorize(G):
    """Factorizes the generator weight to get semantics boundaries.
    Returns:
        A tuple of (semantic_boundaries, eigen_values).
    """
    weights = []
    for layer_name in G.synthesis.layer_names:
        weight = G.synthesis.__getattr__(layer_name).affine.weight.T
        weights.append(weight.cpu().detach().numpy())

    weight = np.concatenate(weights, axis=1).astype(np.float32) # weight.shape -- (512, 4946)

    weight = weight / np.linalg.norm(weight, axis=0, keepdims=True)
    eigen_values, eigen_vectors = np.linalg.eig(weight.dot(weight.T))
    # eigen_vectors.T.shape -- (512, 512)
    # eigen_values.shape -- (512,), descend
    # total = eigen_values.sum() -- 4946.00
    # eigen_values[0:8].sum()/total -- 44.57%
    # eigen_values[0:16].sum()/total -- 55.66%
    # eigen_values[0:32].sum()/total -- 67.13%
    # eigen_values[0:64].sum()/total -- 77.48%
    # eigen_values[0:128].sum()/total -- 85.08%
    # eigen_values[0:256].sum()/total -- 91.92%

    return eigen_vectors.T, eigen_values


def sefa(rand_seeds, output_dir="output"):
    import copy

    # Create directory to store result
    todos.data.mkdir(output_dir)
    device = todos.model.get_device()

    # load model
    D = decoder().to(device)

    boundaries, values = factorize(D)
    distances = np.linspace(-10, 10, 5)

    start_time = time.time()
    progress_bar = tqdm(total=len(rand_seeds))
    for seed in rand_seeds:
        progress_bar.update(1)
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, D.z_dim))
        label = torch.zeros([1,D.c_dim])
        with torch.no_grad():
            w = D.mapping(z.to(device), label.to(device), truncation_psi=0.7)
        codes = w.detach().cpu().numpy()
        del w
        torch.cuda.empty_cache()

        index = 0
        attribute_id = 1

        for d in distances:
            boundary = boundaries[attribute_id:attribute_id+1]
            temp_code = copy.deepcopy(codes)
            temp_code[:,[attribute_id, attribute_id + 1],:] += d * boundary
            with torch.no_grad():
                output_tensor = D.synthesis(torch.from_numpy(temp_code).to(device))
            # model_forward(D, device, temp_code)
            save_tensor(output_tensor, f"{output_dir}/sefa_{seed:06d}_{index:02d}.png")

            del output_tensor
            torch.cuda.empty_cache()

            index += 1

    print("Total spend time:", time.time() - start_time)
