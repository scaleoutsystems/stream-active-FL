"""Script that was used to resize + crop original ZOD images. Mounted into folder called 'ZODCropped' """
import os
import shutil
from PIL import Image, UnidentifiedImageError
from torchvision.transforms.functional import crop

# Custom crop class
class CustomCrop():
    def __init__(self, top=800, left=500, height=800, width=2840):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img):
        return crop(img, self.top, self.left, self.height, self.width)

def process_directory(src_dir, dest_dir, crop_func):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Iterate over all the files and directories in the source directory
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dest_path = os.path.join(dest_dir, item)

        if os.path.isdir(src_path):
            # Recursively process subdirectories
            process_directory(src_path, dest_path, crop_func)
        elif item.endswith('.jpg'):
            # Crop, resize, and save .jpg files
            try:
                with Image.open(src_path) as img:
                    # Apply custom crop before resizing
                    img_cropped = crop_func(img)
                    img_resized = img_cropped.resize((256, 256))
                    img_resized.save(dest_path)
            except UnidentifiedImageError:
                print(f"Skipping corrupted image: {src_path}")
        elif item.endswith('.json') or item.endswith('.npz'):
            # Copy .json and .npz files
            shutil.copy2(src_path, dest_path)

if __name__ == "__main__":
    src_root = "ZOD"
    dest_root = "ZODCropped"

    # Initialize the custom crop function
    custom_crop = CustomCrop()

    # Start processing from the root directory with cropping
    process_directory(src_root, dest_root, custom_crop)
