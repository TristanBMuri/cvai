import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
import argparse
from pathlib import Path
from transformers import CLIPTextModel, CLIPTokenizer
import copy

class MaskedCustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_files = sorted([f for f in self.image_dir.glob("*.png") if not f.name.endswith("_mask.png")])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        base_name = img_path.stem
        caption_path = self.image_dir / f"{base_name}.txt"
        mask_path = self.mask_dir / f"{base_name}_mask.png"
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load mask
        mask = Image.open(mask_path).convert('L')
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = (mask > 0).float()  # binary mask, 1 for foreground, 0 for background
        
        # Load caption
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        
        return {"pixel_values": image, "input_ids": caption, "mask": mask}

class LoRALinearLayer(torch.nn.Module):
    def __init__(self, linear_module, rank=4, alpha=32):
        super().__init__()
        self.linear = linear_module
        self.rank = rank
        self.alpha = alpha
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
        # LoRA weights: (rank, in_features) and (out_features, rank)
        self.lora_A = torch.nn.Parameter(torch.randn(rank, self.linear.in_features) * 0.02)
        self.lora_B = torch.nn.Parameter(torch.zeros(self.linear.out_features, rank))
    def forward(self, x, *args, **kwargs):
        orig_output = self.linear(x, *args, **kwargs)
        # LoRA: project last dimension
        lora_out = F.linear(x, self.lora_A)  # (..., rank)
        lora_out = F.linear(lora_out, self.lora_B)  # (..., out_features)
        return orig_output + lora_out * (self.alpha / self.rank)

def create_lora_unet(unet, rank=4, alpha=32):
    lora_unet = copy.deepcopy(unet)
    for name, module in lora_unet.named_modules():
        if isinstance(module, torch.nn.Linear) and any(x in name for x in ['to_q', 'to_k', 'to_v', 'to_out.0']):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = lora_unet
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            lora_layer = LoRALinearLayer(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_layer)
    return lora_unet

def save_checkpoint(lora_unet, output_dir, epoch, args, accelerator):
    """Save checkpoint after each epoch"""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    state_dict = {}
    unwrapped_unet = accelerator.unwrap_model(lora_unet)
    for name, module in unwrapped_unet.named_modules():
        if isinstance(module, LoRALinearLayer):
            state_dict[f"{name}.lora_A"] = module.lora_A.data.cpu()
            state_dict[f"{name}.lora_B"] = module.lora_B.data.cpu()
    state_dict["lora_config"] = {
        "rank": args.lora_rank,
        "alpha": args.lora_alpha
    }
    torch.save(state_dict, os.path.join(checkpoint_dir, "lora_weights.pt"))
    print(f"Saved checkpoint to {checkpoint_dir}")

def main(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    model_id = "runwayml/stable-diffusion-v1-5"
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    lora_unet = create_lora_unet(unet, rank=args.lora_rank, alpha=args.lora_alpha)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Only train LoRA params
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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = MaskedCustomDataset(
        args.train_data_dir, 
        os.path.join(args.train_data_dir, "masks"),
        transform=transform,
        mask_transform=mask_transform
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    lora_unet, vae, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_unet, vae, text_encoder, optimizer, train_dataloader, lr_scheduler
    )
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    for epoch in range(args.num_train_epochs):
        lora_unet.train()
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == 0 and step < args.resume_step:
                continue
            with accelerator.accumulate(lora_unet):
                latents = vae.encode(batch["pixel_values"].to(accelerator.device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                mask = batch["mask"].to(accelerator.device)
                # Downsample mask to latent size
                mask = torch.nn.functional.interpolate(mask, size=latents.shape[-2:], mode='nearest')
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                text_inputs = tokenizer(
                    batch["input_ids"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                text_input_ids = text_inputs.input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(text_input_ids)[0]
                noise_pred = lora_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                # Only compute loss on unmasked (foreground) regions
                loss = ((noise_pred - noise) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            if global_step >= args.max_train_steps:
                break
        
        # Save checkpoint after each epoch
        if accelerator.is_main_process:
            save_checkpoint(lora_unet, args.output_dir, f"epoch-{epoch}", args, accelerator)
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save final weights
        save_checkpoint(lora_unet, args.output_dir, "final", args, accelerator)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, default="processed_dataset")
    parser.add_argument("--output_dir", type=str, default="tintin_lora")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=30)
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
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()
    main(args) 