import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from IPython.display import display

# Path to the input folder containing images
input_folder = "input"

# Get list of all image files in the folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
image_files.sort()  # Sort alphabetically

def display_image_by_index(index):
    """Display an image from the dataset by its index"""
    if not image_files:
        print(f"No images found in {input_folder}")
        return
    
    # Ensure index is within bounds
    if index < 0 or index >= len(image_files):
        print(f"Index {index} out of bounds. Total images: {len(image_files)}")
        return
    
    # Load and display the image
    image_path = os.path.join(input_folder, image_files[index])
    image = Image.open(image_path)
    
    # Display image details
    print(f"Image {index}/{len(image_files)-1}: {image_files[index]}")
    print(f"Size: {image.size}, Format: {image.format}")
    
    # Display the image
    plt.figure(figsize=(10, 8))
    plt.imshow(np.array(image))
    plt.axis('off')
    plt.show()
    
    return image  # Return the image object if needed for further processing

# Usage example:
# Change this index to view different images from the dataset
# image_index = 0  # First image
# image = display_image_by_index(image_index)

# To view another image, just change image_index and run this cell again
# Examples:
# image_index = 200  # 201st image (0-indexed)
# image = display_image_by_index(image_index) 