import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
import glob
import random

# Configuration
class Config:
    # Model settings
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Training settings
    batch_size = 1
    num_epochs = 10
    learning_rate = 1e-5
    max_train_steps = 1000
    
    # Image settings
    image_size = 512
    
    # Paths
    dataset_path = "VanGogh/VincentVanGogh"
    output_dir = "vangogh_lora"

config = Config()

# Dataset class
class VanGoghDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Recursively find all jpg files
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith(('.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(dirpath, filename))
        
        print(f"Found {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

def main():
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Initialize dataset and dataloader
    dataset = VanGoghDataset(config.dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    # Load models
    vae = AutoencoderKL.from_pretrained(config.model_id, subfolder="vae", use_safetensors=True).to("cuda")
    text_encoder = CLIPTextModel.from_pretrained(config.model_id, subfolder="text_encoder", use_safetensors=True).to("cuda")
    tokenizer = CLIPTokenizer.from_pretrained(config.model_id, subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(config.model_id, subfolder="unet", use_safetensors=True).to("cuda")
    noise_scheduler = DDPMScheduler.from_pretrained(config.model_id, subfolder="scheduler")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Training loop
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
    
    unet.train()
    progress_bar = tqdm(range(config.max_train_steps))

    for epoch in range(config.num_epochs):
        for step, batch in enumerate(dataloader):
            if step >= config.max_train_steps:
                break
                
            # Convert images to latent space
            latents = vae.encode(batch.to("cuda")).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get text embeddings
            prompt = "a painting in the style of van gogh"
            text_input = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to("cuda")
            
            with torch.no_grad():
                text_embeddings = text_encoder(text_input.input_ids)[0]
            
            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # Update weights
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Save the trained model
    os.makedirs(config.output_dir, exist_ok=True)
    unet.save_pretrained(os.path.join(config.output_dir, "unet"))
    print(f"Saved model to {config.output_dir}")

if __name__ == "__main__":
    main() 