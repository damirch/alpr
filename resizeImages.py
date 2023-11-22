from PIL import Image
import os
import numpy as np

# Input and output directories
input_dir = './archive/images/'
output_dir = './archive/images_resized/'

# Desired dimensions
new_width, new_height = 512, 320 #1280, 720

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all files in the input directory
files = os.listdir(input_dir)

for file in files:
    if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
        try:
            # Open image
            img = Image.open(os.path.join(input_dir, file))

            # Resize using PIL
            resized_img = img.resize((new_width, new_height))

            # Convert to numpy array for saving
            img_array = np.array(resized_img)

            # Save the resized image
            output_path = os.path.join(output_dir, file)
            Image.fromarray(img_array).save(output_path)

            print(f"Resized {file} and saved to {output_path}")

        except Exception as e:
            print(f"Failed to process {file}: {e}")
