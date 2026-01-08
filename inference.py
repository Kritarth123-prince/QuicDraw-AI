#!/usr/bin/env python3
"""
Script to generate sketches from a trained model
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Config for inference
class Config:
    img_size = 32
    channels = 1
    timesteps = 1000
    beta_start = 1e-4
    beta_end = 0.02
    hidden_dim = 128
    num_res_blocks = 2

config = Config()

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.time_mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.shortcut(x)

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        h = self.group_norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(batch, channels, height * width).transpose(1, 2)
        k = k.view(batch, channels, height * width).transpose(1, 2)
        v = v.view(batch, channels, height * width).transpose(1, 2)
        
        scale = (channels) ** -0.5
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)
        h = torch.bmm(attn, v).transpose(1, 2).view(batch, channels, height, width)
        
        return self.proj_out(h) + x

class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        time_dim = config.hidden_dim * 4
        self.time_mlp = nn.Sequential(
            TimeEmbedding(config.hidden_dim),
            nn.Linear(config.hidden_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.conv_in = nn.Conv2d(config.channels, config.hidden_dim, 3, padding=1)

        # Encoder blocks
        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(config.hidden_dim, config.hidden_dim, time_dim),
                ResidualBlock(config.hidden_dim, config.hidden_dim, time_dim),
                SelfAttention(config.hidden_dim),
                nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, stride=2, padding=1)
            ),
            nn.Sequential(
                ResidualBlock(config.hidden_dim, config.hidden_dim * 2, time_dim),
                ResidualBlock(config.hidden_dim * 2, config.hidden_dim * 2, time_dim),
                nn.Conv2d(config.hidden_dim * 2, config.hidden_dim * 2, 3, stride=2, padding=1)
            ),
            nn.Sequential(
                ResidualBlock(config.hidden_dim * 2, config.hidden_dim * 4, time_dim),
                ResidualBlock(config.hidden_dim * 4, config.hidden_dim * 4, time_dim),
                nn.Conv2d(config.hidden_dim * 4, config.hidden_dim * 4, 3, stride=2, padding=1)
            ),
        ])

        # Middle block
        self.mid_block = nn.Sequential(
            ResidualBlock(config.hidden_dim * 4, config.hidden_dim * 4, time_dim),
            SelfAttention(config.hidden_dim * 4),
            ResidualBlock(config.hidden_dim * 4, config.hidden_dim * 4, time_dim)
        )

        # Decoder blocks
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(config.hidden_dim * 4, config.hidden_dim * 2, 3, stride=2, padding=1, output_padding=1),
                ResidualBlock(config.hidden_dim * 2, config.hidden_dim * 2, time_dim),
                ResidualBlock(config.hidden_dim * 2, config.hidden_dim * 2, time_dim)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(config.hidden_dim * 2, config.hidden_dim, 3, stride=2, padding=1, output_padding=1),
                ResidualBlock(config.hidden_dim, config.hidden_dim, time_dim),
                ResidualBlock(config.hidden_dim, config.hidden_dim, time_dim)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(config.hidden_dim, config.hidden_dim, 3, stride=2, padding=1, output_padding=1),
                ResidualBlock(config.hidden_dim, config.hidden_dim, time_dim),
                ResidualBlock(config.hidden_dim, config.hidden_dim, time_dim),
                SelfAttention(config.hidden_dim)
            ),
        ])

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, config.hidden_dim),
            nn.SiLU(),
            nn.Conv2d(config.hidden_dim, config.channels, 3, padding=1)
        )

    def forward(self, x, timestep):
        time_emb = self.time_mlp(timestep)
        h = self.conv_in(x)
        skips = [h]
        for block in self.down_blocks:
            for layer in block:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb)
                else:
                    h = layer(h)
            skips.append(h)
        for layer in self.mid_block:
            if isinstance(layer, ResidualBlock):
                h = layer(h, time_emb)
            else:
                h = layer(h)
        for block in self.up_blocks:
            skip = skips.pop()
            h = h + skip
            for layer in block:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb)
                else:
                    h = layer(h)
        return self.conv_out(h)

class DiffusionSampler:
    def __init__(self, config: Config, device):
        self.config = config
        self.device = device
        
        # Noise schedule
        betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]])

    @torch.no_grad()
    def generate(self, model, num_samples=16, steps=None):
        """Generate sketches"""
        if steps is None:
            steps = self.config.timesteps
            
        model.eval()
        
        # Start with noise
        shape = (num_samples, self.config.channels, self.config.img_size, self.config.img_size)
        x = torch.randn(shape, device=self.device)
        
        # Reverse diffusion
        timesteps = list(range(0, self.config.timesteps, self.config.timesteps // steps))[::-1]
        
        for t in tqdm(timesteps, desc="Generating"):
            t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = model(x, t_tensor)
            
            # Remove noise
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            alpha_cumprod_t_prev = self.alphas_cumprod_prev[t]
            
            # Compute previous sample
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            mean = (torch.sqrt(alpha_cumprod_t_prev) * beta_t / (1 - alpha_cumprod_t)) * pred_x0 + \
                   (torch.sqrt(alpha_t) * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)) * x
            
            if t > 0:
                variance = beta_t * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean
        
        return torch.clamp(x, -1, 1)

class SketchGenerator:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = UNet(config).to(self.device)
        self.sampler = DiffusionSampler(config, self.device)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found!")
    
    def generate_sketches(self, num_samples=16, save_path=None):
        """Generate and display sketches"""
        print(f"Generating {num_samples} sketches...")
        
        with torch.no_grad():
            samples = self.sampler.generate(self.model, num_samples)
            
            # Convert to display format
            samples = (samples + 1) / 2  # [-1,1] to [0,1]
            samples = samples.cpu().numpy()
            
            # Display
            rows = int(np.sqrt(num_samples))
            cols = (num_samples + rows - 1) // rows
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
            if rows == 1:
                axes = [axes]
            elif cols == 1:
                axes = [[ax] for ax in axes]
            
            for i in range(num_samples):
                row, col = i // cols, i % cols
                if rows > 1:
                    ax = axes[row][col]
                else:
                    ax = axes[col] if cols > 1 else axes[0]
                    
                ax.imshow(samples[i, 0], cmap='gray', vmin=0, vmax=1)
                ax.axis('off')
            
            # Hide extra subplots
            for i in range(num_samples, rows * cols):
                row, col = i // cols, i % cols
                if rows > 1:
                    axes[row][col].axis('off')
                else:
                    if cols > 1:
                        axes[col].axis('off')
            
            plt.suptitle('Generated Quick Draw Sketches', fontsize=16)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved to {save_path}")
            
            plt.show()
            
        return samples

def main():
    generator = SketchGenerator('model.pth')
    sketches = generator.generate_sketches(num_samples=16, save_path='generated_sketches.png')
    
    print("Generation complete!")

if __name__ == "__main__":
    main()