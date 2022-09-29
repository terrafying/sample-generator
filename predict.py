# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import gc
import json
import math
import os
import random
import sys
from contextlib import contextmanager
from copy import deepcopy
from glob import glob
from pathlib import Path
from re import L

import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import wandb
from cog import BasePredictor, Input, Path
from diffusion import sampling
from einops import rearrange
from google.colab import files
from prefigure.prefigure import get_all_args
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from tqdm import trange

from audio_diffusion.models import DiffusionAttnUnet1D
from audio_diffusion.utils import PadCrop, Stereo

#@title Args
sample_size = 65536 
sample_rate = 48000 
latent_dim = 0              



def report_status(**kwargs):
    status = json.dumps(kwargs)
    print(f"pollen_status: {status}")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")


    def predict(
        self,
        prompt: str,
    ) -> Path:
        """Run a single prediction on the model"""

        # JSON encode {title: "Pimping your prompt", payload: prompt }
        report_status(title="Translating", payload=prompt)
   

        return

#@title Model code
class DiffusionUncond(nn.Module):
    def __init__(self, global_args):
        super().__init__()

        self.diffusion = DiffusionAttnUnet1D(global_args, n_attn_layers = 4)
        self.diffusion_ema = deepcopy(self.diffusion)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

import IPython.display as ipd
import matplotlib.pyplot as plt


def plot_and_hear(audio, sr):
    display(ipd.Audio(audio.cpu().clamp(-1, 1), rate=sr))
    plt.plot(audio.cpu().t().numpy())
  
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

class Object(object):
    pass

args = Object()
args.sample_size = sample_size
args.sample_rate = sample_rate
args.latent_dim = latent_dim

#@title Logging
def get_one_channel(audio_data, channel):
  '''
  Takes a numpy audio array and returns 1 channel
  '''
  # Check if the audio has more than 1 channel 
  if len(audio_data.shape) > 1:
    is_stereo = True      
    if np.argmax(audio_data.shape)==0:
        audio_data = audio_data[:,channel] 
    else:
        audio_data = audio_data[channel,:]
  else:
    is_stereo = False

  return audio_data
