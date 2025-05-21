import os
import json

def create_metadata():
    image_folder = "images_for_gpt"
    metadata_file = "metadata.jsonl"
    
    # Get all image files and their corresponding text files
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        for image_file in image_files:
            txt_file = image_file.replace('.png', '.txt')
            txt_path = os.path.join(image_folder, txt_file)
            
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as txt_f:
                    text = txt_f.read().strip()
                    metadata = {
                        "file_name": image_file,
                        "text": text
                    }
                    f.write(json.dumps(metadata) + '\n')

if __name__ == "__main__":
    create_metadata() 