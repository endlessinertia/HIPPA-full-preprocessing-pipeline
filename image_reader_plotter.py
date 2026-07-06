import os
from pathlib import Path
from collections import defaultdict
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift as nd_shift


class ImageReaderPlotter:

    def __init__(self):
        pass

    @staticmethod
    def group_files_by_prefix(directory_path, extension='h5'):
    
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all .h5 files
        h5_files = list(directory.glob(f'*.{extension}'))
        
        # Group files by MICROTEC prefix
        grouped_files = defaultdict(list)
        
        for file_path in h5_files:
            filename = file_path.name
            
            if filename.startswith('MICROTEC_'):
                parts = filename.split('_')
                if len(parts) >= 2:
                    prefix = f"{parts[0]}_{parts[1]}"
                    grouped_files[prefix].append(filename)
        
        # Convert to regular dict and sort files
        result_dict = {key: sorted(val) for key, val in grouped_files.items()}
        
        return result_dict
    

    @staticmethod
    def read_image(path, image_label, transpose=True):
        h5 = h5py.File(path,'r')        
        if transpose:
            image = h5[image_label]['image']
            return np.transpose(image, (0, 2, 1))
        else:
            image = h5[image_label]
            return np.transpose(image, (0, 1, 2))
        

    @staticmethod
    def save_image(image, path, label):
        with h5py.File(path, "w") as f:
            f.create_dataset(label, data=image)


    @staticmethod
    def crop_image(image, crop_width=300):
        width = image.shape[1]
        start = (width - crop_width) // 2
        end = start + crop_width
        return image[:, start:end]
    

    @staticmethod
    def mirror_image(image):
        return image[:, ::-1, :]
    

    @staticmethod
    def trim_image_channels(image, channels_to_trim=100):
        return image[:, :, channels_to_trim:-channels_to_trim]
    

    @staticmethod
    def dy_shift_image(img, dy_shift):
        return nd_shift(img, shift=(dy_shift, 0, 0), mode='wrap')
    

    @staticmethod
    def plot_time_series_images(images_list, images_names, band_index):

        if len(images_list) != 4 or len(images_names) != 4:
            raise ValueError("Please provide exactly 4 images and 4 image names")
        
        fig, axes = plt.subplots(1, 4, figsize=(40, 25))
        
        for idx, (image, image_name) in enumerate(zip(images_list, images_names)):
            
            axes[idx].imshow(image, cmap='gray')
            axes[idx].set_title(f'{image_name} - Band {band_index}', fontsize=20)
            
            height, width = image.shape
            for y in range(250, height, 250):
                axes[idx].axhline(y=y, color='red', linestyle=':', linewidth=1.5)
            for x in range(100, width, 100):
                axes[idx].axvline(x=x, color='red', linestyle=':', linewidth=1.5)
        
        plt.tight_layout()
        plt.show()
        # plt.savefig(f'MICROTEC_{image_name.split('_')[1]} - Band {band_index} Time Series.png')
        # plt.close()


    @staticmethod
    def plot_aligned_images(ref_img, original_mov_img, shifted_mov_img, apple_name, band_index):

        images_list = [ref_img, original_mov_img, shifted_mov_img]
        images_names = ['Reference Image', 'Original Moving Image', 'Shifted Moving Image']

        fig, axes = plt.subplots(1, 3, figsize=(30, 25))

        for idx, (image, image_name) in enumerate(zip(images_list, images_names)):

            axes[idx].imshow(image, cmap='gray')
            axes[idx].set_title(f'{apple_name} - {image_name} - Band {band_index}', fontsize=20)

            height, width = image.shape
            for y in range(250, height, 250):
                axes[idx].axhline(y=y, color='red', linestyle=':', linewidth=1.5)
            for x in range(100, width, 100):
                axes[idx].axvline(x=x, color='red', linestyle=':', linewidth=1.5)

        plt.tight_layout()
        plt.show()

    
    @staticmethod
    def plot_image(image, image_name, band_index):
        title = f'{image_name} - Band {band_index}'
        
        plt.figure(figsize=(30, 20))
        plt.imshow(image[:, :, band_index], cmap='gray')
        plt.title(title)

        height, width, _ = image.shape
        for y in range(250, height, 250):
            plt.axhline(y=y, color='red', linestyle=':', linewidth=1.5)
        for x in range(100, width, 100):
            plt.axvline(x=x, color='red', linestyle=':', linewidth=1.5)

        plt.tight_layout()
        plt.show()
        # plt.savefig(f'{title}.png')
        # plt.close()


    @staticmethod
    def plot_time_series_rgb_images(images_list, images_names):

        if len(images_list) != 4 or len(images_names) != 4:
            raise ValueError("Please provide exactly 4 images and 4 image names")
        
        fig, axes = plt.subplots(1, 4, figsize=(40, 25))
        
        for idx, (image, image_name) in enumerate(zip(images_list, images_names)):
            
            axes[idx].imshow(image) 
            axes[idx].set_title(f'{image_name}', fontsize=20)
            
            height, width, _ = image.shape  # RGB shape is (H, W, 3), so slice to first 2 dims
            for y in range(250, height, 250):
                axes[idx].axhline(y=y, color='red', linestyle=':', linewidth=1.5)
            for x in range(100, width, 100):
                axes[idx].axvline(x=x, color='red', linestyle=':', linewidth=1.5)
        
        plt.tight_layout()
        plt.show()