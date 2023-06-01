from contextlib import contextmanager
from prefigure.prefigure import get_all_args
from copy import deepcopy
import math
from pathlib import Path
from google.colab import files

import wandb
import os
import signal
import sys
import gc

from diffusion import sampling
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from tqdm import trange
from einops import rearrange

import torchaudio
from audio_diffusion.models import DiffusionAttnUnet1D
import numpy as np

import random
import matplotlib.pyplot as plt
import IPython.display as ipd
from audio_diffusion.utils import Stereo, PadCrop
from glob import glob

# @title Model code


class DiffusionUncond(nn.Module):
    def __init__(self, global_args):
        super().__init__()

        self.diffusion = DiffusionAttnUnet1D(global_args, n_attn_layers=4)
        self.diffusion_ema = deepcopy(self.diffusion)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)


def load_to_device(path, sr):
    audio, file_sr = torchaudio.load(path)
    if sr != file_sr:
        audio = torchaudio.transforms.Resample(file_sr, sr)(audio)
    audio = audio.to(device)
    return audio


def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma ** 2) ** 0.5
    return alpha_sigma_to_t(alpha, sigma)


def t_to_alpha_sigma(t):
    """Returns the scaling factors for the clean image and for the noise, given
    a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2


# @title Args
sample_size = 65536
sample_rate = 48000
latent_dim = 0


class Object(object):
    pass


args = Object()
args.sample_size = sample_size
args.sample_rate = sample_rate
args.latent_dim = latent_dim

# @title Logging


def get_one_channel(audio_data, channel):
    '''
    Takes a numpy audio array and returns 1 channel
    '''
    # Check if the audio has more than 1 channel
    if len(audio_data.shape) > 1:
        is_stereo = True
        if np.argmax(audio_data.shape) == 0:
            audio_data = audio_data[:, channel]
        else:
            audio_data = audio_data[channel, :]
    else:
        is_stereo = False

    return audio_data


def log_audio_to_wandb(
    generated, model_name, custom_ckpt_path, steps, batch_size, sample_rate, sample_size,
    generated_all=None, channel=0, original_sample=None, gen_type='new_sounds', noise_level=None, sample_length_mult=None, file_path=None
):

    print('\nSaving your audio generations to Weights & Biases...')

    # Get model name
    if model_name == "custom":
        wandb_model_name = custom_ckpt_path
    else:
        wandb_model_name = model_name

    # Create config to log to wandb
    wandb_config = {
        "model": model_name,
        "steps": steps,
        "batch_size": batch_size,
        "sample_rate": sample_rate,
        "sample_size": sample_size,
        "channel": channel,
        "gen_type": gen_type,
        "noise_level": noise_level,
        "sample_length_mult": sample_length_mult,
        "file_path": file_path
    }

    # Create a new wandb run
    wandb.init(project='harmonai-audio-gen', config=wandb_config)
    wandb_run_url = wandb.run.get_url()

    # Create a Weights & Biases Table
    audio_generations_table = wandb.Table(columns=['audio', 'steps', 'model', 'batch_size',
                                                   'sample_rate', 'sample_size', 'duration'])

    # Add each individual generated sample to a wandb Table
    for idx, g in enumerate(generated.cpu().numpy()):

        # Check if the audio has more than 1 channel
        if idx == 0:
            if len(g.shape) > 1:
                stereo = True
            else:
                stereo = False

        if stereo:
            g = g[channel]

        duration = np.max(g.shape) / sample_rate
        wandb_audio = wandb.Audio(
            g, sample_rate=sample_rate, caption=wandb_model_name)
        audio_generations_table.add_data(wandb_audio, steps, wandb_model_name, batch_size,
                                         sample_rate, sample_size, duration)

    # Log the samples Tables and finish the wandb run
    wandb.log({f'{gen_type}/harmonai_generations': audio_generations_table})

    # Log the combined samples in another wandb Table
    if generated_all is not None:
        g_all = get_one_channel(generated_all, channel)
        duration_all = np.max(g_all.shape) / sample_rate
        audio_all_generations_table = wandb.Table(columns=['audio', 'steps', 'model', 'batch_size',
                                                           'sample_rate', 'sample_size', 'duration'])
        wandb_all_audio = wandb.Audio(
            g_all.cpu().numpy(), sample_rate=sample_rate, caption=wandb_model_name)
        audio_all_generations_table.add_data(wandb_all_audio, steps, wandb_model_name, batch_size,
                                             sample_rate, sample_size, duration_all)
        wandb.log(
            {f'{gen_type}/all_harmonai_generations': audio_all_generations_table})

    if original_sample is not None:
        original_sample = get_one_channel(original_sample, channel)
        audio_original_sample_table = wandb.Table(
            columns=['audio', 'file_path'])
        wandb_original_audio = wandb.Audio(
            original_sample, sample_rate=sample_rate)
        audio_original_sample_table.add_data(wandb_original_audio, file_path)
        wandb.log({f'{gen_type}/original_sample': audio_original_sample_table})

    wandb.finish()

    print(
        f'Your audio generations are saved in Weights & Biases here: {wandb_run_url}\n')


# @markdown How many audio clips to create
batch_size = 4  # @param {type:"number"}

# @markdown Number of steps (100 is a good start, more steps trades off speed for quality)
steps = 120  # @param {type:"number"}

# @markdown Multiplier on the default sample length from the model, allows for longer audio clips at the expense of VRAM
sample_length_mult = 1  # @param {type:"number"}

# @markdown Check the box below to save your generated audio to [Weights & Biases](https://www.wandb.ai/site)
save_new_generations_to_wandb = True

# @markdown Check the box below to skip this section when running all cells
skip_for_run_all = False  # @param {type: "boolean"}

effective_length = sample_length_mult * args.sample_size

if not skip_for_run_all:
    torch.cuda.empty_cache()
    gc.collect()

    # Generate random noise to sample from
    noise = torch.randn([batch_size, 2, effective_length]).to(device)

    generated = sample(model_fn, noise, steps, sampler_type)

    # Hard-clip the generated audio
    generated = generated.clamp(-1, 1)

    # Put the demos together
    generated_all = rearrange(generated, 'b d n -> d (b n)')

    print("All samples")
    for ix, gen_sample in enumerate(generated):
        print(f'sample #{ix + 1}')

    # If Weights & Biases logging enabled, save generations
    if save_new_generations_to_wandb:
        # Check if logged in to wandb
        try:
            import netrc
            netrc.netrc().hosts['api.wandb.ai']

            log_audio_to_wandb(generated, model_name, custom_ckpt_path, steps, batch_size,
                               args.sample_rate, args.sample_size, generated_all=generated_all)
        except:
            print("Not logged in to Weights & Biases, please tick the `save_to_wandb` box at the top of this notebook and run that cell again to log in to Weights & Biases first")

else:
    print("Skipping section, uncheck 'skip_for_run_all' to enable")
