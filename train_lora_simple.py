import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import json
from tqdm.auto import tqdm

class CustomDataset(Dataset):
    def __init__(self, metadata_file, image_dir, tokenizer):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.metadata = []
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.metadata.append(json.loads(line))
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_path = os.path.join(self.image_dir, item['file_name'])
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Tokenize text
        text = item['text']
        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=77,
            return_tensors='pt'
        )
        
        return {
            'pixel_values': image,
            'input_ids': tokens['input_ids'].squeeze()
        }

def main():
    # Load base model
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cuda")

    # Create dataset and dataloader
    dataset = CustomDataset("metadata.jsonl", "images_for_gpt", pipe.tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize LoRA attention processors
    lora_attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else pipe.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = pipe.unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")].split(".")[0])
            hidden_size = list(reversed(pipe.unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")].split(".")[0])
            hidden_size = pipe.unet.config.block_out_channels[block_id]
        
        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=4
        )

    pipe.unet.set_attn_processor(lora_attn_procs)

    # Training parameters
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-4)
    num_epochs = 50

    # Create output directory
    os.makedirs("lora_output", exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to GPU
            batch = {k: v.to("cuda") for k, v in batch.items()}
            
            # Forward pass
            noise = torch.randn_like(batch['pixel_values'])
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (batch['pixel_values'].shape[0],), device="cuda")
            noisy_images = pipe.scheduler.add_noise(batch['pixel_values'], noise, timesteps)
            
            # Get model prediction
            noise_pred = pipe.unet(noisy_images, timesteps, batch['input_ids']).sample
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0 and step == len(dataloader) - 1:
                checkpoint_dir = os.path.join("lora_output", f"checkpoint-{epoch+1}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                pipe.unet.save_pretrained(checkpoint_dir)
                print(f"\nSaved checkpoint to {checkpoint_dir}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    # Save final model
    pipe.unet.save_pretrained(os.path.join("lora_output", "final"))
    print("\nTraining complete! Final weights saved to 'lora_output/final'")

if __name__ == "__main__":
    main() 