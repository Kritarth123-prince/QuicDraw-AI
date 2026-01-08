#!/usr/bin/env python3
"""
Mini Diffusion Model for Google Quick Draw Dataset
Optimized for Google Colab T4 GPU

This script implements a lightweight diffusion model for generating sketches
using the Google Quick Draw dataset, with optimizations for T4 GPU performance.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
import gzip
import json
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Install required packages for Colab
def install_requirements():
    """Install required packages if not already present"""
    import subprocess
    import sys

    packages = ['requests', 'numpy', 'matplotlib', 'torch', 'torchvision', 'tqdm']
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Call installation function
install_requirements()

# Configuration
class Config:
    # Model parameters
    img_size = 32  # Small size for T4 GPU efficiency
    channels = 1   # Grayscale sketches
    timesteps = 1000
    beta_start = 1e-4
    beta_end = 0.02

    # Training parameters
    batch_size = 64  # Optimized for T4 16GB VRAM
    learning_rate = 2e-4
    epochs = 50
    save_interval = 10

    # Model architecture
    hidden_dim = 128
    num_res_blocks = 2
    attention_levels = [16]  # Apply attention at 16x16 resolution

    # Dataset parameters
    max_categories = 10  # Limit categories for faster training
    max_samples_per_category = 1000

    # Optimization flags for T4
    mixed_precision = True
    gradient_checkpointing = True
    pin_memory = True
    num_workers = 2

config = Config()

# Quick Draw Dataset Handler
class QuickDrawDataset(Dataset):
    def __init__(self, config: Config):
        self.config = config
        self.data = []
        self.categories = [
            'cat', 'dog', 'car', 'house', 'tree',
            'flower', 'fish', 'bird', 'airplane', 'bicycle'
        ][:config.max_categories]

        print("Downloading and processing Quick Draw data...")
        self._load_data()

    def _load_data(self):
        """Download and process Quick Draw dataset"""
        base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

        for category in tqdm(self.categories, desc="Loading categories"):
            url = f"{base_url}{category}.npy"

            try:
                # Download category data
                response = requests.get(url)
                if response.status_code == 200:
                    # Save temporarily
                    temp_file = f"/tmp/{category}.npy"
                    with open(temp_file, 'wb') as f:
                        f.write(response.content)

                    # Load and process
                    category_data = np.load(temp_file)

                    # Limit samples per category
                    if len(category_data) > self.config.max_samples_per_category:
                        indices = np.random.choice(len(category_data),
                                                 self.config.max_samples_per_category,
                                                 replace=False)
                        category_data = category_data[indices]

                    # Reshape to 28x28 and resize to target size
                    category_data = category_data.reshape(-1, 28, 28)

                    # Convert to target size
                    resized_data = []
                    for img in category_data:
                        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
                        resized = F.interpolate(img_tensor.unsqueeze(0),
                                              size=(self.config.img_size, self.config.img_size),
                                              mode='bilinear', align_corners=False)
                        resized_data.append(resized.squeeze().numpy())

                    self.data.extend(resized_data)

                    # Clean up
                    os.remove(temp_file)

            except Exception as e:
                print(f"Error loading {category}: {e}")
                continue

        self.data = np.array(self.data)
        print(f"Loaded {len(self.data)} samples from {len(self.categories)} categories")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        # Normalize to [-1, 1]
        image = (image / 255.0) * 2.0 - 1.0
        return torch.FloatTensor(image).unsqueeze(0)  # Add channel dimension

# Diffusion Model Components
def get_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
    """Linear beta schedule for diffusion"""
    return torch.linspace(beta_start, beta_end, timesteps)

def get_alpha_schedule(betas):
    """Compute alpha values from betas"""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas, alphas_cumprod

# U-Net Architecture for T4 optimization
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

        # Reshape for attention computation
        q = q.view(batch, channels, height * width).transpose(1, 2)
        k = k.view(batch, channels, height * width).transpose(1, 2)
        v = v.view(batch, channels, height * width).transpose(1, 2)

        # Scaled dot-product attention
        scale = (channels) ** -0.5
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)
        h = torch.bmm(attn, v).transpose(1, 2).view(batch, channels, height, width)

        return self.proj_out(h) + x

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

class UNet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Time embedding
        time_dim = config.hidden_dim * 4
        self.time_mlp = nn.Sequential(
            TimeEmbedding(config.hidden_dim),
            nn.Linear(config.hidden_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Encoder
        self.conv_in = nn.Conv2d(config.channels, config.hidden_dim, 3, padding=1)

        self.down_blocks = nn.ModuleList([
            # 32x32 -> 16x16
            self._make_down_block(config.hidden_dim, config.hidden_dim, time_dim, True),
            # 16x16 -> 8x8
            self._make_down_block(config.hidden_dim, config.hidden_dim * 2, time_dim, False),
            # 8x8 -> 4x4
            self._make_down_block(config.hidden_dim * 2, config.hidden_dim * 4, time_dim, False),
        ])

        # Middle
        self.mid_block = nn.Sequential(
            ResidualBlock(config.hidden_dim * 4, config.hidden_dim * 4, time_dim),
            SelfAttention(config.hidden_dim * 4),
            ResidualBlock(config.hidden_dim * 4, config.hidden_dim * 4, time_dim)
        )

        # Decoder
        self.up_blocks = nn.ModuleList([
            # 4x4 -> 8x8
            self._make_up_block(config.hidden_dim * 4, config.hidden_dim * 2, time_dim, False),
            # 8x8 -> 16x16
            self._make_up_block(config.hidden_dim * 2, config.hidden_dim, time_dim, False),
            # 16x16 -> 32x32
            self._make_up_block(config.hidden_dim, config.hidden_dim, time_dim, True),
        ])

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, config.hidden_dim),
            nn.SiLU(),
            nn.Conv2d(config.hidden_dim, config.channels, 3, padding=1)
        )

    def _make_down_block(self, in_channels, out_channels, time_dim, has_attention):
        layers = []
        for _ in range(self.config.num_res_blocks):
            layers.append(ResidualBlock(in_channels, out_channels, time_dim))
            in_channels = out_channels

        if has_attention:
            layers.append(SelfAttention(out_channels))

        layers.append(nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1))
        return nn.Sequential(*layers)

    def _make_up_block(self, in_channels, out_channels, time_dim, has_attention):
        layers = [nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)]

        for _ in range(self.config.num_res_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, time_dim))

        if has_attention:
            layers.append(SelfAttention(out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, timestep):
        time_emb = self.time_mlp(timestep)

        # Encoder
        h = self.conv_in(x)
        skip_connections = [h]

        for block in self.down_blocks:
            for layer in block:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb)
                else:
                    h = layer(h)
            skip_connections.append(h)

        # Middle
        for layer in self.mid_block:
            if isinstance(layer, ResidualBlock):
                h = layer(h, time_emb)
            else:
                h = layer(h)

        # Decoder
        for block in self.up_blocks:
            skip = skip_connections.pop()
            h = h + skip  # Skip connection
            for layer in block:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb)
                else:
                    h = layer(h)

        return self.conv_out(h)

# Diffusion Process
class DDPM:
    def __init__(self, config: Config, device):
        self.config = config
        self.device = device

        # Set up noise schedule
        self.betas = get_beta_schedule(config.timesteps, config.beta_start, config.beta_end).to(device)
        self.alphas, self.alphas_cumprod = get_alpha_schedule(self.betas)

        # Precompute values for efficiency
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Add alpha_cumprod_prev for proper sampling
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_prev = alphas_cumprod_prev

    def add_noise(self, x_start, t, noise=None):
        """Add noise to clean images"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def sample_timesteps(self, batch_size):
        """Sample random timesteps for training"""
        return torch.randint(0, self.config.timesteps, (batch_size,), device=self.device)

def visualize_training_data(dataset, num_samples=16):
    """Visualize some training data samples"""
    print("Visualizing training data...")

    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        sample = dataset[indices[i]]
        # Convert from [-1, 1] to [0, 1] for display
        img = (sample.squeeze().numpy() + 1) / 2
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')

    plt.suptitle('Training Data Samples', fontsize=16)
    plt.tight_layout()
    plt.show()

# Training Loop
def train_model(config: Config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Initialize dataset and dataloader
    dataset = QuickDrawDataset(config)

    # Visualize some training data first
    visualize_training_data(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    # Initialize model and diffusion
    model = UNet(config).to(device)
    diffusion = DDPM(config, device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

    # Mixed precision scaler
    scaler = GradScaler() if config.mixed_precision else None

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Training loop
    model.train()
    global_step = 0
    best_loss = float('inf')

    for epoch in range(config.epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            batch = batch.to(device, non_blocking=True)
            batch_size = batch.shape[0]

            # Sample noise and timesteps
            noise = torch.randn_like(batch)
            t = diffusion.sample_timesteps(batch_size)

            # Add noise to images
            x_t = diffusion.add_noise(batch, t, noise)

            optimizer.zero_grad()

            if config.mixed_precision and scaler:
                with autocast():
                    # Predict noise
                    predicted_noise = model(x_t, t)
                    loss = F.mse_loss(predicted_noise, noise)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Predict noise
                predicted_noise = model(x_t, t)
                loss = F.mse_loss(predicted_noise, noise)

                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        avg_epoch_loss = epoch_loss / len(dataloader)
        scheduler.step()

        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, 'quickdraw_diffusion_best.pth')

        # Save model checkpoint and generate samples
        if (epoch + 1) % config.save_interval == 0 or epoch == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, f'quickdraw_diffusion_epoch_{epoch+1}.pth')

            # Generate sample images
            model.eval()
            generate_samples(model, diffusion, device, epoch+1)
            model.train()

    return model, diffusion

# Sampling/Generation Functions
@torch.no_grad()
def sample_ddpm(model, diffusion, shape, device, num_inference_steps=None):
    """Generate samples using DDPM sampling"""
    if num_inference_steps is None:
        num_inference_steps = diffusion.config.timesteps

    model.eval()

    # Start from pure noise
    x = torch.randn(shape, device=device)

    # Use all timesteps in reverse order
    timesteps = list(range(diffusion.config.timesteps))[::-1]

    for i, t in enumerate(tqdm(timesteps, desc="Generating")):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # Predict noise
        predicted_noise = model(x, t_tensor)

        # Get coefficients for this timestep
        alpha_t = diffusion.alphas[t]
        alpha_cumprod_t = diffusion.alphas_cumprod[t]
        beta_t = diffusion.betas[t]

        # Compute mean of reverse process
        alpha_cumprod_t_prev = diffusion.alphas_cumprod_prev[t]

        # Compute x_{t-1} mean
        pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)  # Clamp to valid range

        # Compute mean of q(x_{t-1} | x_t, x_0)
        mean = (torch.sqrt(alpha_cumprod_t_prev) * beta_t / (1 - alpha_cumprod_t)) * pred_x0 + \
               (torch.sqrt(alpha_t) * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)) * x

        if t > 0:
            # Compute variance
            variance = beta_t * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(variance) * noise
        else:
            x = mean

    return x

def generate_samples(model, diffusion, device, epoch=None):
    """Generate and display sample images"""
    print(f"Generating samples...")

    with torch.no_grad():
        # Generate samples
        samples = sample_ddpm(
            model, diffusion,
            (16, diffusion.config.channels, diffusion.config.img_size, diffusion.config.img_size),
            device,
            num_inference_steps=None  # Use all timesteps for better quality
        )

        print(f"Generated samples shape: {samples.shape}")
        print(f"Sample value range: [{samples.min().item():.3f}, {samples.max().item():.3f}]")

        # Convert to numpy and denormalize
        samples = torch.clamp(samples, -1, 1)  # Ensure proper range
        samples = (samples + 1) / 2  # Convert from [-1, 1] to [0, 1]
        samples = samples.cpu().numpy()

        print(f"After denormalization range: [{samples.min():.3f}, {samples.max():.3f}]")

        # Create grid of images
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            img = samples[i, 0]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

        if epoch:
            plt.suptitle(f'Generated Sketches - Epoch {epoch}', fontsize=16)
        else:
            plt.suptitle('Generated Sketches', fontsize=16)

        plt.tight_layout()
        plt.show()

        if epoch:
            plt.savefig(f'generated_samples_epoch_{epoch}.png', dpi=150, bbox_inches='tight')

    return samples

# Main execution function
def main():
    """Main training function"""
    print("🎨 Quick Draw Diffusion Model Training")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Image size: {config.img_size}x{config.img_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Mixed precision: {config.mixed_precision}")
    print(f"  Categories: {config.max_categories}")
    print("=" * 50)

    try:
        # Train the model
        model, diffusion = train_model(config)

        print("\n🎉 Training completed successfully!")

        # Generate final samples
        print("Generating final samples...")
        generate_samples(model, diffusion, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

# Interactive generation function for post-training use
def generate_new_sketches(model_path, num_samples=16):
    """Load trained model and generate new sketches"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = UNet(config).to(device)
    diffusion = DDPM(config, device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded model from {model_path}")
    print(f"Training was at epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")

    # Generate samples
    generate_samples(model, diffusion, device)

if __name__ == "__main__":
    # For Colab: Automatically start training
    print("Starting training automatically...")
    main()

    # Instructions for manual generation after training
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nTo generate new sketches after training, use:")
    print("generate_new_sketches('quickdraw_diffusion_epoch_50.pth')")
    print("\nOr modify the config and retrain:")
    print("config.epochs = 100  # Train for more epochs")
    print("config.batch_size = 32  # Reduce if memory issues")
    print("main()  # Start training again")