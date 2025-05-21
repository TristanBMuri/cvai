import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from accelerate import Accelerator
from tqdm.auto import tqdm

# Define the dataset class
class ImageDescriptionDataset(Dataset):
    def __init__(self, image_folder, tokenizer, max_length=77):
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_file)
        txt_path = image_path.replace('.png', '.txt')
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Load description
        with open(txt_path, 'r', encoding='utf-8') as f:
            description = f.read().strip()
        
        # Tokenize description
        tokens = self.tokenizer(
            description,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'pixel_values': image,
            'input_ids': tokens['input_ids'].squeeze()
        }

def main():
    # Initialize accelerator
    accelerator = Accelerator()

    # Load model
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    
    # Prepare dataset
    dataset = ImageDescriptionDataset("images_for_gpt", pipe.tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

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
    
    # Optimizer
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-4)

    # Move to device
    pipe = pipe.to(accelerator.device)
    train_dataloader = accelerator.prepare(train_dataloader)
    optimizer = accelerator.prepare(optimizer)

    # Training loop
    num_epochs = 50
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            # Forward pass
            loss = pipe(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids']
            ).loss
            
            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            checkpoint_dir = f"checkpoint_epoch_{epoch+1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            pipe.unet.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved to {checkpoint_dir}")

    # Save final model
    pipe.unet.save_pretrained("lora_weights_final")
    print("Training complete! Final weights saved to 'lora_weights_final'")

if __name__ == "__main__":
    main() 