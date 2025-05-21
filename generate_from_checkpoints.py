import os
import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from pathlib import Path
import argparse
from tqdm import tqdm
import copy
from PIL import Image

class LoRALinearLayer(torch.nn.Module):
    def __init__(self, linear_module, rank=4, alpha=32):
        super().__init__()
        self.linear = linear_module
        self.rank = rank
        self.alpha = alpha
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
        self.lora_A = None
        self.lora_B = None

    def forward(self, x, *args, **kwargs):
        orig_output = self.linear(x, *args, **kwargs)
        if self.lora_A is not None and self.lora_B is not None:
            # LoRA: project last dimension
            lora_out = F.linear(x, self.lora_A)  # (..., rank)
            lora_out = F.linear(lora_out, self.lora_B)  # (..., out_features)
            return orig_output + lora_out * (self.alpha / self.rank)
        return orig_output

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

def load_lora_weights(unet, checkpoint_path):
    """Load LoRA weights from checkpoint"""
    state_dict = torch.load(os.path.join(checkpoint_path, "lora_weights.pt"))
    lora_config = state_dict.pop("lora_config")
    
    # Apply LoRA weights to UNet
    for name, module in unet.named_modules():
        if isinstance(module, LoRALinearLayer):
            if f"{name}.lora_A" in state_dict:
                module.lora_A = state_dict[f"{name}.lora_A"].to(unet.device)
                module.lora_B = state_dict[f"{name}.lora_B"].to(unet.device)
    
    return lora_config

@torch.no_grad()
def main(args):
    # Load models
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load all components
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # Prepare UNet with LoRA
    unet = create_lora_unet(unet)
    unet.eval()
    text_encoder.eval()
    vae.eval()
    
    # Base prompt that will be used for all generations
    base_prompt = args.prompt + ", Tintin comics, Hergé style, ligne claire illustration"
    
    # Get all checkpoint directories
    checkpoint_dirs = sorted([
        d for d in os.listdir(args.checkpoint_dir)
        if d.startswith("checkpoint-epoch-")
    ])
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate images for each checkpoint
    for checkpoint_dir in tqdm(checkpoint_dirs, desc="Processing checkpoints"):
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_dir)
        
        # Load LoRA weights
        lora_config = load_lora_weights(unet, checkpoint_path)
        
        # Create checkpoint-specific output directory
        checkpoint_output_dir = output_dir / checkpoint_dir
        checkpoint_output_dir.mkdir(exist_ok=True)
        
        # Encode text
        text_input = tokenizer(
            [base_prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        
        # Generate 3 images for this checkpoint
        for i in range(3):
            # Set different seed for each image
            generator = torch.Generator(device=device).manual_seed(args.seed + i)
            
            # Create random latent noise
            latents = torch.randn(
                (1, unet.config.in_channels, args.height // 8, args.width // 8),
                generator=generator,
                device=device
            )
            
            # Set timesteps
            noise_scheduler.set_timesteps(args.num_inference_steps)
            
            # Denoising loop
            for t in noise_scheduler.timesteps:
                # expand the latents for classifier free guidance
                latent_model_input = latents
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                
                # predict the noise residual
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            
            # scale and decode the image latents with vae
            latents = 1 / vae.config.scaling_factor * latents
            image = vae.decode(latents).sample
            
            # Convert to PIL image
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).round().astype("uint8")
            image = Image.fromarray(image)
            
            # Save image
            image.save(checkpoint_output_dir / f"image_{i}.png")
            
            # Save prompt used
            with open(checkpoint_output_dir / f"image_{i}_prompt.txt", "w") as f:
                f.write(base_prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="tintin_lora",
                      help="Directory containing the checkpoints")
    parser.add_argument("--output_dir", type=str, default="generated_images",
                      help="Directory to save generated images")
    parser.add_argument("--prompt", type=str, required=True,
                      help="Base prompt to use for generation")
    parser.add_argument("--seed", type=int, default=42,
                      help="Base seed for generation")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                      help="Number of denoising steps")
    parser.add_argument("--height", type=int, default=512,
                      help="Height of generated images")
    parser.add_argument("--width", type=int, default=512,
                      help="Width of generated images")
    
    args = parser.parse_args()
    main(args) 