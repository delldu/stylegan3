# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generator architecture from the paper
"Alias-Free Generative Adversarial Networks"."""

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import time

def profiled_function(fn):
    def decorator(*args, **kwargs):
        torch.cuda.synchronize()        
        start_time = time.time()
        with torch.autograd.profiler.record_function(fn.__name__):
            y = fn(*args, **kwargs)
        torch.cuda.synchronize()        
        spend_time = time.time() - start_time
        if spend_time > 0.01:
            print(f"{fn.__name__} spend time: {spend_time:0.5f}")
        return y
    decorator.__name__ = fn.__name__
    return decorator

def bias_lrelu_act(x, b=None):
    if b is not None:
        x = x + b.reshape([-1 if i == 1 else 1 for i in range(x.ndim)])

    x = F.leaky_relu(x, 0.2)
    x = x * np.sqrt(2)
    return x.clamp(-256, 256)


def upfir2d(x, f, up=1, padding=[0, 0], gain=1):
    """Up Sample FIR filter for 2D"""
    batch_size, num_channels, in_height, in_width = x.shape

    # if len(padding) == 2:
    #     padx, pady = padding
    #     padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding

    # Upsample by inserting zeros.
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = F.pad(x, [0, up - 1, 0, 0, 0, up - 1])
    x = x.reshape([batch_size, num_channels, in_height * up, in_width * up])

    # Pad or crop.
    x = F.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
    x = x[:, :, max(-pady0, 0) : x.shape[2] - max(-pady1, 0), max(-padx0, 0) : x.shape[3] - max(-padx1, 0)]

    # Setup filter.
    f = f * (gain ** (f.ndim / 2))
    f = f.to(x.dtype)
    f = f.flip(list(range(f.ndim)))
    # f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    f = f[None, None].repeat([num_channels, 1] + [1] * f.ndim)

    # Convolve with the filter.
    if f.ndim == 4:
        # x.size() -- [1, 3, 1024, 1024]
        # f.size() -- [3, 1, 1, 1]
        # num_channels -- 3
        x = F.conv2d(input=x, weight=f, groups=num_channels)
        # ==> x.size() -- [1, 3, 1024, 1024]
    else:
        # x.size() -- [1, 1024, 93, 93]
        # f.size(): [1024, 1, 12]-->[1024, 1, 1, 12]
        # num_channels -- 1024
        x = F.conv2d(input=x, weight=f.unsqueeze(2), groups=num_channels)
        # ==> x.size() -- [1, 1024, 93, 82]
        x = F.conv2d(input=x, weight=f.unsqueeze(3), groups=num_channels)
        # ==> x.size() -- [1, 1024, 82, 82]
    return x


def dnfir2d(x, f, down=1):
    """Down Sample FIR filter for 2D"""

    batch_size, num_channels, in_height, in_width = x.shape
    # Setup filter.
    f = f.to(x.dtype)
    f = f.flip(list(range(f.ndim)))
    # f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    f = f[None, None].repeat([num_channels, 1] + [1] * f.ndim)

    if f.ndim == 4:
        x = F.conv2d(input=x, weight=f, groups=num_channels)
    else:
        x = F.conv2d(input=x, weight=f.unsqueeze(2), groups=num_channels)
        x = F.conv2d(input=x, weight=f.unsqueeze(3), groups=num_channels)

    # Downsample by throwing away pixels.
    return x[:, :, ::down, ::down]

# @profiled_function
def filtered_lrelu(x, fu=None, fd=None, b=None, up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2):
    # Slow implement
    x = x + b.reshape([-1 if i == 1 else 1 for i in range(x.ndim)])
    x = upfir2d(x=x, f=fu, up=up, padding=padding, gain=up ** 2)  # Upsample.
    x = F.leaky_relu(x, slope) * gain
    x = dnfir2d(x=x, f=fd, down=down)  # Downsample.

    return x


def modulated_conv2d(
    x,  # Input tensor: [batch_size, in_channels, in_height, in_width]
    w,  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
    s,  # Style tensor: [batch_size, in_channels]
    demodulate=True,  # Apply weight demodulation?
    padding=0,  # Padding: int or [padH, padW]
    input_gain=None,  # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
):
    if x.is_cuda:
        # Reduce memory !!!
        x = x.to(torch.float16)

    batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape

    if demodulate:
        w = w * w.square().mean([1, 2, 3], keepdim=True).rsqrt()
        s = s * s.square().mean().rsqrt()

    # Modulate weights.
    w = w.unsqueeze(0)  # [NOIkk]
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # [NOIkk]

    # Demodulate weights.
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
        w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [NOIkk]

    # if input_gain is not None:
    #     input_gain = input_gain.expand(batch_size, in_channels)  # [NI]
    #     w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # [NOIkk]
    input_gain = input_gain.expand(batch_size, in_channels)  # [NI]
    w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # [NOIkk]

    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = F.conv2d(input=x, weight=w.to(x.dtype), padding=padding, groups=batch_size)

    return x.reshape(batch_size, -1, *x.shape[2:])


class FullyConnectedLayer(nn.Module):
    def __init__(
        self,
        in_features,  # Number of input features.
        out_features,  # Number of output features.
        activation="linear",  # Activation function: 'relu', 'lrelu', etc.
        bias=True,  # Apply additive bias before the activation function?
        lr_multiplier=1,  # Learning rate multiplier.
        weight_init=1,  # Initial standard deviation of the weight tensor.
        bias_init=0,  # Initial value of the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

        # self = FullyConnectedLayer(in_features=512, out_features=4, activation=linear)
        # in_features = 512
        # out_features = 4
        # activation = 'linear'
        # bias = True
        # lr_multiplier = 1
        # weight_init = 0
        # bias_init = array([1., 0., 0., 0.], dtype=float32)

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        # b = self.bias
        # if b is not None:
        #     b = b.to(x.dtype)
        #     if self.bias_gain != 1:
        #         b = b * self.bias_gain
        b = self.bias * self.bias_gain

        if self.activation == "linear":
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_lrelu_act(x, b)
        return x

    def extra_repr(self):
        return f"in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}"


class MappingNetwork(nn.Module):
    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality.
        c_dim,  # Conditioning label (C) dimensionality, 0 = no labels.
        w_dim,  # Intermediate latent (W) dimensionality.
        num_ws,  # Number of intermediate latents to output.
        num_layers=2,  # Number of mapping layers.
        lr_multiplier=0.01,  # Learning rate multiplier for the mapping layers.
    ):
        super().__init__()

        # z_dim = 512
        # c_dim = 0
        # w_dim = 512
        # num_ws = 16
        # num_layers = 2
        # lr_multiplier = 0.01

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers

        # Construct layers.
        features = [self.z_dim] + [self.w_dim] * self.num_layers
        # features ---  [512, 512, 512]
        for idx, in_features, out_features in zip(range(num_layers), features[:-1], features[1:]):
            layer = FullyConnectedLayer(in_features, out_features, activation="lrelu", lr_multiplier=lr_multiplier)
            setattr(self, f"fc{idx}", layer)
        self.register_buffer("w_avg", torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None):
        if truncation_cutoff is None:
            truncation_cutoff = self.num_ws
            
        # Embed, normalize, and concatenate inputs.
        x = z.to(torch.float32)
        x = x * (x.square().mean(1, keepdim=True) + 1e-8).rsqrt()

        # Execute layers.
        for idx in range(self.num_layers):
            x = getattr(self, f"fc{idx}")(x)

        # Broadcast and apply truncation.
        x = x.unsqueeze(1).repeat([1, self.num_ws, 1])
        if truncation_psi != 1:
            x[:, : truncation_cutoff] = self.w_avg.lerp(x[:, : truncation_cutoff], truncation_psi)

        return x

    def extra_repr(self):
        return f"z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}"


class SynthesisInput(nn.Module):
    def __init__(
        self,
        w_dim,  # Intermediate latent (W) dimensionality.
        channels,  # Number of output channels.
        size,  # Output spatial size: int or [width, height].
        sampling_rate,  # Output sampling rate.
        bandwidth,  # Output bandwidth.
    ):
        super().__init__()

        # self = SynthesisInput(
        #   w_dim=512, channels=1024, size=[36, 36],
        #   sampling_rate=16, bandwidth=2
        #   (affine): FullyConnectedLayer(in_features=512, out_features=4, activation=linear)
        # )
        # w_dim = 512
        # channels = 1024
        # size = 36
        # sampling_rate = 16.0
        # bandwidth = 2.0

        self.w_dim = w_dim
        self.channels = channels
        self.size = np.broadcast_to(np.asarray(size), [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth

        # Draw random frequencies from uniform 2D disc.
        freqs = torch.randn([self.channels, 2])
        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= bandwidth
        phases = torch.rand([self.channels]) - 0.5

        # Setup parameters and buffers.
        self.weight = torch.nn.Parameter(torch.randn([self.channels, self.channels]))
        self.affine = FullyConnectedLayer(w_dim, 4, weight_init=0, bias_init=[1, 0, 0, 0])
        self.register_buffer("transform", torch.eye(3, 3))  # User-specified inverse transform wrt. resulting image.
        self.register_buffer("freqs", freqs)
        self.register_buffer("phases", phases)

    def forward(self, w):
        # Introduce batch dimension.
        transforms = self.transform.unsqueeze(0)  # [batch, row, col]
        freqs = self.freqs.unsqueeze(0)  # [batch, channel, xy]
        phases = self.phases.unsqueeze(0)  # [batch, channel]

        # Apply learned transformation.
        t = self.affine(w)  # t = (r_c, r_s, t_x, t_y)
        t = t / t[:, :2].norm(dim=1, keepdim=True)  # t' = (r'_c, r'_s, t'_x, t'_y)
        m_r = (
            torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1])
        )  # Inverse rotation wrt. resulting image.
        m_r[:, 0, 0] = t[:, 0]  # r'_c
        m_r[:, 0, 1] = -t[:, 1]  # r'_s
        m_r[:, 1, 0] = t[:, 1]  # r'_s
        m_r[:, 1, 1] = t[:, 0]  # r'_c
        m_t = (
            torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1])
        )  # Inverse translation wrt. resulting image.
        m_t[:, 0, 2] = -t[:, 2]  # t'_x
        m_t[:, 1, 2] = -t[:, 3]  # t'_y
        transforms = (
            m_r @ m_t @ transforms
        )  # First rotate resulting image, then translate, and finally apply user-specified transform.

        # Transform frequencies.
        phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transforms[:, :2, :2]

        # Dampen out-of-band frequencies that may occur due to the user-specified transform.
        amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)

        # Construct sampling grid.
        theta = torch.eye(2, 3, device=w.device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grids = torch.nn.functional.affine_grid(
            theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False
        )

        # Compute Fourier features.
        x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(
            3
        )  # [batch, height, width, channel]
        x = x + phases.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (np.pi * 2))
        x = x * amplitudes.unsqueeze(1).unsqueeze(2)

        # Apply trainable mapping.
        weight = self.weight / np.sqrt(self.channels)
        x = x @ weight.t()

        # Ensure correct shape.
        return x.permute(0, 3, 1, 2)  # [batch, channel, height, width]

    def extra_repr(self):
        return "\n".join(
            [
                f"w_dim={self.w_dim:d}, channels={self.channels:d}, size={list(self.size)},",
                f"sampling_rate={self.sampling_rate:g}, bandwidth={self.bandwidth:g}",
            ]
        )


class SynthesisLayer(nn.Module):
    def __init__(
        self,
        w_dim,  # Intermediate latent (W) dimensionality.
        is_torgb,  # Is this the final ToRGB layer?
        is_critically_sampled,  # Does this layer use critical sampling?
        use_fp16,  # Does this layer use FP16?
        # Input & output specifications.
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        in_size,  # Input spatial size: int or [width, height].
        out_size,  # Output spatial size: int or [width, height].
        in_sampling_rate,  # Input sampling rate (s).
        out_sampling_rate,  # Output sampling rate (s).
        in_cutoff,  # Input cutoff frequency (f_c).
        out_cutoff,  # Output cutoff frequency (f_c).
        in_half_width,  # Input transition band half-width (f_h).
        out_half_width,  # Output Transition band half-width (f_h).
        # Hyperparameters.
        conv_kernel=3,  # Convolution kernel size. Ignored for final the ToRGB layer.
        filter_size=6,  # Low-pass filter size relative to the lower resolution when up/downsampling.
        lrelu_upsampling=2,  # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
        use_radial_filters  = False,    # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
        conv_clamp=256,  # Clamp the output to [-X, +X], None = disable clamping.
        config_r = False
    ):
        super().__init__()

        # for config R
        if config_r:
            conv_kernel = 1
            use_radial_filters = True

        self.w_dim = w_dim
        self.is_torgb = is_torgb
        self.is_critically_sampled = is_critically_sampled
        self.use_fp16 = use_fp16
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.conv_kernel = 1 if is_torgb else conv_kernel
        self.conv_clamp = conv_clamp

        # Setup parameters and buffers.
        self.affine = FullyConnectedLayer(self.w_dim, self.in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(
            torch.randn([self.out_channels, self.in_channels, self.conv_kernel, self.conv_kernel])
        )
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        self.register_buffer("magnitude_ema", torch.ones([]))

        # Design upsampling filter.
        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.up_factor == self.tmp_sampling_rate
        self.up_taps = filter_size * self.up_factor if self.up_factor > 1 and not self.is_torgb else 1
        self.register_buffer(
            "up_filter",
            self.design_lowpass_filter(
                numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width * 2, fs=self.tmp_sampling_rate
            ),
        )

        # Design downsampling filter.
        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.down_factor == self.tmp_sampling_rate
        self.down_taps = filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
        self.down_radial = use_radial_filters and not self.is_critically_sampled
        self.register_buffer(
            "down_filter",
            self.design_lowpass_filter(
                numtaps=self.down_taps,
                cutoff=self.out_cutoff,
                width=self.out_half_width * 2,
                fs=self.tmp_sampling_rate,
                radial=self.down_radial,
            ),
        )

        # Compute padding.
        pad_total = (self.out_size - 1) * self.down_factor + 1  # Desired output size before downsampling.
        pad_total -= (self.in_size + self.conv_kernel - 1) * self.up_factor  # Input size after upsampling.
        pad_total += self.up_taps + self.down_taps - 2  # Size reduction caused by the filters.
        pad_lo = (pad_total + self.up_factor) // 2
        # Shift sample locations according to the symmetric interpretation (Appendix C.3).
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

    def forward(self, x, w, noise_mode="random"):
        assert noise_mode in ["random", "const", "none"]  # unused

        if self.up_filter is None:
            self.up_filter = torch.ones([1, 1], dtype=torch.float32).to(x.device)

        if self.down_filter is None:
            self.down_filter = torch.ones([1, 1], dtype=torch.float32).to(x.device)

        # Track input magnitude.
        input_gain = self.magnitude_ema.rsqrt()

        # Execute affine layer.
        styles = self.affine(w)
        if self.is_torgb:
            weight_gain = 1 / np.sqrt(self.in_channels * (self.conv_kernel ** 2))
            styles = styles * weight_gain

        # Execute modulated conv2d.
        x = modulated_conv2d(
            x=x,
            w=self.weight,
            s=styles,
            padding=self.conv_kernel - 1,
            demodulate=(not self.is_torgb),
            input_gain=input_gain,
        )

        # Execute bias, filtered leaky ReLU, and clamping.
        gain = 1 if self.is_torgb else np.sqrt(2)
        slope = 1 if self.is_torgb else 0.2
        return filtered_lrelu(
                x=x,
                fu=self.up_filter,
                fd=self.down_filter,
                b=self.bias.to(x.dtype),
                up=self.up_factor,
                down=self.down_factor,
                padding=self.padding,
                gain=gain,
                slope=slope,
            )

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
        assert numtaps >= 1
        # numtaps = 12
        # cutoff = 2.0
        # width = 12.0
        # fs = 32
        # radial = False

        # Identity filter.
        if numtaps == 1:
            return None

        # Separable Kaiser low-pass filter.
        if not radial:
            f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return torch.as_tensor(f, dtype=torch.float32)

        # Radially symmetric jinc-based filter.
        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)

        return torch.as_tensor(f, dtype=torch.float32)

    def extra_repr(self):
        return "\n".join(
            [
                f"w_dim={self.w_dim:d}, is_torgb={self.is_torgb},",
                f"is_critically_sampled={self.is_critically_sampled}, use_fp16={self.use_fp16},",
                f"in_sampling_rate={self.in_sampling_rate:g}, out_sampling_rate={self.out_sampling_rate:g},",
                f"in_cutoff={self.in_cutoff:g}, out_cutoff={self.out_cutoff:g},",
                f"in_half_width={self.in_half_width:g}, out_half_width={self.out_half_width:g},",
                f"in_size={list(self.in_size)}, out_size={list(self.out_size)},",
                f"in_channels={self.in_channels:d}, out_channels={self.out_channels:d}",
            ]
        )


class SynthesisNetwork(nn.Module):
    def __init__(
        self,
        w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # Output image resolution.
        img_channels,  # Number of color channels.
        channel_base=32768,  # Overall multiplier for the number of channels.
        channel_max=512,  # Maximum number of channels in any layer.
        num_layers=14,  # Total number of layers, excluding Fourier features and ToRGB.
        num_critical=2,  # Number of critically sampled layers at the end.
        first_cutoff=2,  # Cutoff frequency of the first layer (f_{c,0}).
        first_stopband=2 ** 2.1,  # Minimum stopband of the first layer (f_{t,0}).
        last_stopband_rel=2 ** 0.3,  # Minimum stopband of the last layer, expressed relative to the cutoff.
        margin_size=10,  # Number of additional pixels outside the image.
        output_scale=0.25,  # Scale factor for the output image.
        num_fp16_res=4,  # Use FP16 for the N highest resolutions.
        config_r = False,
        **layer_kwargs,  # Arguments for SynthesisLayer.
    ):
        super().__init__()

        # For config R
        if config_r:
            channel_base *= 2
            channel_max *= 2

        # w_dim = 512
        # img_resolution = 1024
        # img_channels = 3
        # channel_base = 65536
        # channel_max = 1024
        # num_layers = 14
        # num_critical = 2
        # first_cutoff = 2
        # first_stopband = 4.2870938501451725
        # last_stopband_rel = 1.2311444133449163
        # margin_size = 10
        # output_scale = 0.25
        # num_fp16_res = 4
        # layer_kwargs = {}

        self.w_dim = w_dim
        self.num_ws = num_layers + 2
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_layers = num_layers
        self.num_critical = num_critical
        self.margin_size = margin_size
        self.output_scale = output_scale
        self.num_fp16_res = num_fp16_res

        # Geometric progression of layer cutoffs and min. stopbands.
        last_cutoff = self.img_resolution / 2  # f_{c,N}

        last_stopband = last_cutoff * last_stopband_rel  # f_{t,N}
        exponents = np.minimum(np.arange(self.num_layers + 1) / (self.num_layers - self.num_critical), 1)

        cutoffs = first_cutoff * (last_cutoff / first_cutoff) ** exponents  # f_c[i]
        stopbands = first_stopband * (last_stopband / first_stopband) ** exponents  # f_t[i]

        # Compute remaining layer parameters.
        sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands * 2, self.img_resolution))))  # s[i]
        half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs  # f_h[i]
        sizes = sampling_rates + self.margin_size * 2
        sizes[-2:] = self.img_resolution
        channels = np.rint(np.minimum((channel_base / 2) / cutoffs, channel_max))

        if img_resolution == 256:
            # Adjust model channels 
            channels[7] = int(channels[7]/1.414)
            for i in range(8, self.num_layers):
                channels[i] = channels[i] // 2

        channels[-1] = self.img_channels

        # Construct layers.
        self.input = SynthesisInput(
            w_dim=self.w_dim,
            channels=int(channels[0]),
            size=int(sizes[0]),
            sampling_rate=sampling_rates[0],
            bandwidth=cutoffs[0],
        )
        self.layer_names = []

        # print("self.num_layers --- ", self.num_layers) -- 14
        for idx in range(self.num_layers + 1):
            prev = max(idx - 1, 0)
            is_torgb = idx == self.num_layers
            is_critically_sampled = idx >= self.num_layers - self.num_critical
            use_fp16 = sampling_rates[idx] * (2 ** self.num_fp16_res) > self.img_resolution
            layer = SynthesisLayer(
                w_dim=self.w_dim,
                is_torgb=is_torgb,
                is_critically_sampled=is_critically_sampled,
                use_fp16=use_fp16,
                in_channels=int(channels[prev]),
                out_channels=int(channels[idx]),
                in_size=int(sizes[prev]),
                out_size=int(sizes[idx]),
                in_sampling_rate=int(sampling_rates[prev]),
                out_sampling_rate=int(sampling_rates[idx]),
                in_cutoff=cutoffs[prev],
                out_cutoff=cutoffs[idx],
                in_half_width=half_widths[prev],
                out_half_width=half_widths[idx],
                config_r = config_r,
                **layer_kwargs,
            )
            name = f"L{idx}_{layer.out_size[0]}_{layer.out_channels}"
            setattr(self, name, layer)
            self.layer_names.append(name)

    def forward(self, ws):
        # misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        ws = ws.to(torch.float32).unbind(dim=1)

        # Execute layers.
        x = self.input(ws[0])
        for name, w in zip(self.layer_names, ws[1:]):
            x = getattr(self, name)(x, w)

        if self.output_scale != 1:
            x = x * self.output_scale

        # Ensure correct shape and dtype.
        # misc.assert_shape(x, [None, self.img_channels, self.img_resolution, self.img_resolution])
        x = (x + 1.0)/2.0
        # return x.to(torch.float32).clamp(0, 1.0)
        return x.to(torch.float32)

    def extra_repr(self):
        return "\n".join(
            [
                f"w_dim={self.w_dim:d}, num_ws={self.num_ws:d},",
                f"img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},",
                f"num_layers={self.num_layers:d}, num_critical={self.num_critical:d},",
                f"margin_size={self.margin_size:d}, num_fp16_res={self.num_fp16_res:d}",
            ]
        )


class Generator(nn.Module):
    def __init__(
        self,
        z_dim=512,  # Input latent (Z) dimensionality.
        c_dim=0,  # Conditioning label (C) dimensionality.
        w_dim=512,  # Intermediate latent (W) dimensionality.
        img_resolution=1024,  # Output resolution.
        img_channels=3,  # Number of output color channels.
        mapping_kwargs={},  # Arguments for MappingNetwork.
        config_r = False,
        **synthesis_kwargs,  # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(
            w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, config_r=config_r, **synthesis_kwargs
        )
        self.num_ws = self.synthesis.num_ws
        # self.num_ws -- 16
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi)
        img = self.synthesis(ws, **synthesis_kwargs)

        # pp z.size() -- torch.Size([1, 512])
        # pp ws.size() -- torch.Size([1, 16, 512])

        # pp img.size() -- torch.Size([1, 3, 1024, 1024])
        return img
