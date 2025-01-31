import math
import torch
import torch.optim as optim
import torch.nn as nn
import os
import logging
import time
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import yaml
import argparse
import shutil
import torchvision.utils as tvu
import torchvision.transforms.functional as F
import h5py
import itertools
import sys
sys.path.append('../') 
from models.dpm_utils import *

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def get_timestep_embedding_adm(
    timesteps: torch.Tensor,
    num_channels: int,
    max_positions: int = 10000,
    endpoint: bool = False,
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: A tensor of timesteps.
        num_channels: The number of channels to generate.
        max_positions: The maximum number of positions to generate embeddings for.
        endpoint: If True, include the endpoint in the range.

    Returns:
        A tensor of shape `(timesteps.shape[0], num_channels)` containing the embeddings.
    """
    freqs = torch.arange(start=0, end=num_channels//2, dtype=torch.float32, device=timesteps.device)
    freqs = freqs / (num_channels // 2 - (1 if endpoint else 0))
    freqs = (1 / max_positions) ** freqs
    emb = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([emb.cos(), emb.sin()], dim=1)
    return emb

class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float,
        temb_channels: int = 64,
    ) -> None:
        """
        Initialize a ResnetBlock.

        Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels. If None, in_channels is used.
        dropout: The dropout rate.
        temb_channels: The number of channels in the time embedding.
        """
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.LayerNorm(in_channels)
        
        self.dense1 = torch.nn.Linear(in_channels, out_channels)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.dense2 = torch.nn.Linear(out_channels, out_channels)
        
        if self.in_channels != self.out_channels:
            self.shortcut = torch.nn.Linear(in_channels,
                                                 out_channels)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.dense1(h)

        h = h + self.temb_proj(nonlinearity(temb))

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.dense2(h)

        if self.in_channels != self.out_channels:
            x = self.shortcut(x)

        return x+h

    
class Model(nn.Module):
    def __init__(self, config: Any) -> None:
        """
        Initialize the Model.

        Args:
            config: A configuration object containing model hyperparameters.
                - model (object): Model configuration.
                    - ch (int): Number of channels.
                    - ch_mult (Tuple[int, ...]): Channel multipliers for each resolution level.
                    - num_res_blocks (int): Number of residual blocks.
                    - dropout (float): Dropout rate.
                    - type (str): Type of the model, e.g., 'bayesian'.
                    - c_noise (Optional[bool]): Whether to use noise conditioning.
                - data (object): Data configuration.
                    - input_size (int): Input size for the model.
                - diffusion (object): Diffusion configuration.
                    - num_diffusion_timesteps (int): Number of diffusion timesteps.
        """
        super().__init__()
        self.config = config
        ch: int = config.model.ch
        out_ch: int = config.data.input_size
        ch_mult: Tuple[int, ...] = tuple(config.model.ch_mult)
        num_res_blocks: int = config.model.num_res_blocks
        dropout: float = config.model.dropout
        in_channels: int = config.data.input_size
        input_size: int = config.data.input_size
        num_timesteps: int = config.diffusion.num_diffusion_timesteps
        c_noise: bool = config.model.c_noise if config.model.c_noise is not None else False

        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))

        self.ch: int = ch
        self.temb_ch: int = self.ch
        self.num_resolutions: int = len(ch_mult)
        self.num_res_blocks: int = num_res_blocks
        self.resolution: int = input_size
        self.in_channels: int = in_channels
        self.c_noise: bool = c_noise

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # downsampling
        self.layer_in = torch.nn.Linear(in_channels, self.ch)

        curr_res: int = input_size
        in_ch_mult: Tuple[int, ...] = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in: Optional[int] = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.LayerNorm(block_in)
        self.layer_out = torch.nn.Linear(block_in, out_ch)

    def forward(self, x, t):

        if self.c_noise:
            t = torch.log(t)
        
        # timestep embedding
        # print('ch:', self.ch)
        temb = get_timestep_embedding_adm(t, self.ch)
        "timesteps, embedding_dim"
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        h = self.layer_in(x)
        # print('inside the forward:', x.shape, h.shape, temb.shape)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                # hs.append(h)
            # if i_level != self.num_resolutions-1:
            #     hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        # h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)
        
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block[i_block](h, temb)
            # if i_level != 0:
            #     h = self.up[i_level].upsample(h)
            
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.layer_out(h)
        # print('at the end of forawrd h:', h.shape)
        return h
        
        
def generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et = model(xt, t)
            x0_t = (xt - et * extract_into_tensor((1 - at).sqrt(), xt.shape)) / extract_into_tensor(at.sqrt(), xt.shape)
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = extract_into_tensor(at_next.sqrt(), x0_t.shape) * x0_t + extract_into_tensor(c1, x0_t.shape) * torch.randn_like(x) + extract_into_tensor(c2, x0_t.shape) * et
            xs.append(xt_next.to('cpu'))
            
    xs = [xs_.to('cpu').numpy() for xs_ in xs]
    # xs = np.concatenate(xs, axis=0)
    
    x0_preds = [x0_preds_.to('cpu').numpy() for x0_preds_ in x0_preds]
    # x0_preds = np.concatenate(x0_preds, axis=0)
    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to(x.device)

            output = model(x, t.float())
            e = output

            x0_from_e = extract_into_tensor((1.0 / at).sqrt(), x.shape) * x - extract_into_tensor((1.0 / at - 1).sqrt(), e.shape)* e
            x0_from_e = torch.clamp(x0_from_e, -3, 3)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                extract_into_tensor((atm1.sqrt() * beta_t), x0_from_e.shape) * x0_from_e + extract_into_tensor(((1 - beta_t).sqrt() * (1 - atm1)), x.shape) * x
            ) / extract_into_tensor((1.0 - at), x.shape)
            
            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1)
            logvar = beta_t.log()
            sample = mean + extract_into_tensor(mask, noise.shape) * extract_into_tensor(torch.exp(0.5 * logvar), noise.shape) * noise
            xs.append(sample.to('cpu'))
            
    xs = [xs_.to('cpu').numpy() for xs_ in xs]
    # xs = np.concatenate(xs, axis=0)
    
    x0_preds = [x0_preds_.to('cpu').numpy() for x0_preds_ in x0_preds]
    # x0_preds = np.concatenate(x0_preds, axis=0)
    
    return xs, x0_preds