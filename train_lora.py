import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
import argparse
from pathlib import Path
from transformers import CLIPTextModel, CLIPTokenizer
import copy
import types
import shutil

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_files = []
        
        # Recursively find all PNG files
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.png'):
                    self.image_files.append(os.path.join(root, file))
        
        if not self.image_files:
            raise ValueError(f"No PNG files found in {image_dir} or its subdirectories")
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        caption_path = f"{os.path.splitext(img_path)[0]}.txt"
        
        # Load image and handle transparency
        image = Image.open(img_path).convert('RGBA')
        # Create a white background
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        # Composite the image over the white background
        image = Image.alpha_composite(background, image)
        image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Load caption
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
            
        return {"pixel_values": image, "input_ids": caption}

class LoRALinearLayer(torch.nn.Module):
    def __init__(self, linear_module, rank=4, alpha=32):
        super().__init__()
        self.linear = linear_module
        self.rank = rank
        self.alpha = alpha
        
        # Freeze the original weights
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
        
        # Initialize LoRA weights
        self.lora_A = torch.nn.Parameter(
            torch.randn(self.linear.in_features, rank) * 0.02
        )
        self.lora_B = torch.nn.Parameter(
            torch.zeros(rank, self.linear.out_features)
        )
    
    def forward(self, x):
        # Regular forward
        orig_output = self.linear(x)
        
        # LoRA forward
        if x.dim() > 0:  # Skip if input is 0D
            lora_output = F.linear(x, self.lora_A)
            lora_output = F.linear(lora_output, self.lora_B)
            return orig_output + lora_output * (self.alpha / self.rank)
        return orig_output

def create_lora_unet(unet, rank=4, alpha=32):
    # Create a copy of the unet
    lora_unet = copy.deepcopy(unet)
    
    # Dictionary to hold all LoRA layers
    lora_layers = {}
    
    # Add trainable LoRA adapters to the UNet
    for name, module in lora_unet.named_modules():
        if isinstance(module, torch.nn.Linear) and any(x in name for x in ['to_q', 'to_k', 'to_v', 'to_out.0']):
            # Get parent module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = lora_unet
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            
            # Create LoRA layer
            lora_layer = LoRALinearLayer(module, rank=rank, alpha=alpha)
            
            # Replace the layer
            setattr(parent, child_name, lora_layer)
            
            # Add to dictionary
            lora_layers[name] = lora_layer
    
    return lora_unet

def main(args):
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # Load pretrained model and tokenizer
    model_id = "runwayml/stable-diffusion-v1-5"
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    
    # Freeze the original model
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Create LoRA UNet
    lora_unet = create_lora_unet(unet, rank=args.lora_rank, alpha=args.lora_alpha)
    
    # Setup noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # Setup optimizer - only train the LoRA parameters
    lora_params = []
    for module in lora_unet.modules():
        if isinstance(module, LoRALinearLayer):
            lora_params.extend([module.lora_A, module.lora_B])
    
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Setup dataset
    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    train_dataset = CustomDataset(args.train_data_dir, transform=transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    
    # Calculate max_train_steps if not provided
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * len(train_dataloader)
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # Prepare everything with accelerator
    lora_unet, vae, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_unet, vae, text_encoder, optimizer, train_dataloader, lr_scheduler
    )
    
    # Training loop
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    global_step = 0
    for epoch in range(args.num_train_epochs):
        lora_unet.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == 0 and step < args.resume_step:
                continue
                
            with accelerator.accumulate(lora_unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(accelerator.device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                
                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get the text embedding for conditioning
                text_inputs = tokenizer(
                    batch["input_ids"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                text_input_ids = text_inputs.input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(text_input_ids)[0]
                
                # Predict the noise residual
                noise_pred = lora_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Compute loss
                loss = F.mse_loss(noise_pred, noise)
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            # Checks if the accelerator has performed an optimization step
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            
            if global_step >= args.max_train_steps:
                break
    
    # Save the model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save only the LoRA parameters
        os.makedirs(args.output_dir, exist_ok=True)
        state_dict = {}
        
        # Unwrap the model
        unwrapped_unet = accelerator.unwrap_model(lora_unet)
        
        # Save LoRA parameters
        for name, module in unwrapped_unet.named_modules():
            if isinstance(module, LoRALinearLayer):
                state_dict[f"{name}.lora_A"] = module.lora_A.data.cpu()
                state_dict[f"{name}.lora_B"] = module.lora_B.data.cpu()
        
        # Save configuration
        state_dict["lora_config"] = {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha
        }
        
        torch.save(state_dict, os.path.join(args.output_dir, "lora_weights.pt"))
        
def collect_images_and_captions(source_dir, target_dir):
    """
    Collect all PNG images and their corresponding text captions from source_dir
    (including subdirectories) and copy them to target_dir in a flat structure.
    """
    # Create target directory if it doesn't exist
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True, parents=True)
    
    # Counter to avoid filename conflicts
    counter = 1
    
    # Walk through all subdirectories
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.png'):
                # Get the full path of the image
                img_path = os.path.join(root, file)
                
                # Get corresponding caption file path
                caption_path = f"{os.path.splitext(img_path)[0]}.txt"
                
                # Check if caption file exists
                if not os.path.exists(caption_path):
                    print(f"Warning: No caption found for {img_path}")
                    continue
                
                # Create new filenames with counter to avoid conflicts
                new_img_name = f"image_{counter:04d}.png"
                new_caption_name = f"image_{counter:04d}.txt"
                
                # Copy files to target directory
                shutil.copy2(img_path, target_path / new_img_name)
                shutil.copy2(caption_path, target_path / new_caption_name)
                
                print(f"Copied {img_path} -> {new_img_name}")
                counter += 1
    
    print(f"Collected {counter-1} image-caption pairs in {target_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, default="panels_no_bubbles")
    parser.add_argument("--output_dir", type=str, default="tintin_lora")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--resume_step", type=int, default=0)
    # LoRA specific arguments
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    args = parser.parse_args()
    main(args)
