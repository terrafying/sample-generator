# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import gc
import hashlib
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
from urllib.parse import urlparse

import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
import torchaudio
import wandb
from cog import BasePredictor, Input, Path
from diffusion import sampling
from einops import rearrange
from prefigure.prefigure import get_all_args
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from tqdm import trange

from audio_diffusion.models import DiffusionAttnUnet1D
from audio_diffusion.utils import PadCrop, Stereo

#@title Args

sample_rate = 48000 
latent_dim = 0              



def wget(url, outputdir):
    # Using the !wget command instead of the subprocess to get the loading bar
    os.system(f"wget {url} -O {outputdir}")
    # res = subprocess.run(['wget', url, '-P', f'{outputdir}'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    # print(res)

def report_status(**kwargs):
    status = json.dumps(kwargs)
    print(f"pollen_status: {status}")


# #@markdown Number of steps (100 is a good start, more steps trades off speed for quality)
# steps = 100 #@param {type:"number"}

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        self.loaded_model_fn = None
        self.loaded_model_name = None
        os.system("ls -l /models")
    def predict(
        self,
        model_name: str = Input(description="Model", default = "glitch-440k", choices=["glitch-440k",  "jmann-large-580k", "maestro-150k", "unlocked-250k"]),
        length: str = Input(description="Number of seconds to generate", default="8", choices=["2","4","8","12","16"]),
        batch_size: int = Input(description="How many samples to generate", default=2),
        steps: int = Input(description="Number of steps, higher numbers will give more refined output but will take longer. The maximum is 150.", default=100),
    ) -> Path:
        """Run a single prediction on the model"""

        # JSON encode {title: "Pimping your prompt", payload: prompt }
        #report_status(title="Translating", payload=prompt)

        sample_size = sample_rate * int(length)
        args = Object()
        args.sample_size = sample_size
        args.sample_rate = sample_rate
        args.latent_dim = latent_dim

        #@title Create the model
        model_path = "/models"

        model_info = models_map[model_name]
        args.sample_size = model_info["sample_size"]
        args.sample_rate = model_info["sample_rate"]

        if self.loaded_model_name != model_name:
            download_model(model_name,0,model_path)
            ckpt_path = f'{model_path}/{get_model_filename(model_name)}'        
            print("Creating the model...")
            model = DiffusionUncond(args)
            model.load_state_dict(torch.load(ckpt_path)["state_dict"])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.requires_grad_(False).to(device)
            # # Remove non-EMA
            del model.diffusion

            self.loaded_model_fn = model.diffusion_ema
            print("Model created")
        
        model_fn = self.loaded_model_fn



        #@markdown Check the box below to save your generated audio to [Weights & Biases](https://www.wandb.ai/site)
        save_new_generations_to_wandb = False #@param {type: "boolean"}


        torch.cuda.empty_cache()
        gc.collect()

        # Generate random noise to sample from
        noise = torch.randn([batch_size, 2, args.sample_size]).to(device)

        t = torch.linspace(1, 0, steps + 1, device=device)[:-1]
        step_list = get_crash_schedule(t)

        # Generate the samples from the noise
        generated = sampling.iplms_sample(model_fn, noise, step_list, {})

        # Hard-clip the generated audio
        generated = generated.clamp(-1, 1)

        print("All samples")
        # plot_and_hear(generated_all, args.sample_rate)
        samples = []
        for ix, gen_sample in enumerate(generated):
            print(f'sample #{ix + 1}')
            #audio = ipd.Audio(gen_sample.cpu(), rate=args.sample_rate)
            print(gen_sample.shape)
            samples.append(gen_sample)       
        else:
            print("Skipping section, uncheck 'skip_for_run_all' to enable")

        # concatenate the samples
        samples = torch.cat(samples, dim=1)
        # save to disk (format is c n)
        soundfile.write(f'/tmp/sample.wav', samples.permute(1,0).cpu().numpy(), args.sample_rate)
        return Path(f"/tmp/sample.wav")



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

print("hey")



def get_model_filename(diffusion_model_name):
    model_uri = models_map[diffusion_model_name]['uri_list'][0]
    model_filename = os.path.basename(urlparse(model_uri).path)
    return model_filename

def download_model(diffusion_model_name, uri_index=0, model_path='/models'):
    if diffusion_model_name != 'custom':
        model_filename = get_model_filename(diffusion_model_name)
        model_local_path = os.path.join(model_path, model_filename)


        if not models_map[diffusion_model_name]['downloaded']:
            for model_uri in models_map[diffusion_model_name]['uri_list']:
                wget(model_uri, model_local_path)
                with open(model_local_path, "rb") as f:
                  bytes = f.read() 
                  hash = hashlib.sha256(bytes).hexdigest()
                  print(f'SHA: {hash}')
                if os.path.exists(model_local_path):
                    models_map[diffusion_model_name]['downloaded'] = True
                    return
                else:
                    print(f'{diffusion_model_name} model download from {model_uri} failed. Will try any fallback uri.')
            print(f'{diffusion_model_name} download failed.')


models_map = {

    "glitch-440k": {'downloaded': True,
                         'sha': "48caefdcbb7b15e1a0b3d08587446936302535de74b0e05e0d61beba865ba00a", 
                         'uri_list': ["https://model-server.zqevans2.workers.dev/gwf-440k.ckpt"],
                         'sample_rate': 48000,
                         'sample_size': 65536
                         },
    "jmann-small-190k": {'downloaded': False,
                         'sha': "1e2a23a54e960b80227303d0495247a744fa1296652148da18a4da17c3784e9b", 
                         'uri_list': ["https://model-server.zqevans2.workers.dev/jmann-small-190k.ckpt"],
                         'sample_rate': 48000,
                         'sample_size': 65536
                         },
    "jmann-large-580k": {'downloaded': True,
                         'sha': "6b32b5ff1c666c4719da96a12fd15188fa875d6f79f8dd8e07b4d54676afa096", 
                         'uri_list': ["https://model-server.zqevans2.workers.dev/jmann-large-580k.ckpt"],
                         'sample_rate': 48000,
                         'sample_size': 131072
                         },
    "maestro-150k": {'downloaded': True,
                         'sha': "49d9abcae642e47c2082cec0b2dce95a45dc6e961805b6500204e27122d09485", 
                         'uri_list': ["https://model-server.zqevans2.workers.dev/maestro-uncond-150k.ckpt"],
                         'sample_rate': 16000,
                         'sample_size': 65536
                         },
    "unlocked-250k": {'downloaded': True,
                         'sha': "af337c8416732216eeb52db31dcc0d49a8d48e2b3ecaa524cb854c36b5a3503a", 
                         'uri_list': ["https://model-server.zqevans2.workers.dev/unlocked-uncond-250k.ckpt"],
                         'sample_rate': 16000,
                         'sample_size': 65536
                         },
    "honk-140k": {'downloaded': False,
                         'sha': "a66847844659d287f55b7adbe090224d55aeafdd4c2b3e1e1c6a02992cb6e792", 
                         'uri_list': ["https://model-server.zqevans2.workers.dev/honk-140k.ckpt"],
                         'sample_rate': 16000,
                         'sample_size': 65536
                         },
}
