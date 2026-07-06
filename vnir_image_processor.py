from reflectance_normalizer import ReflectanceNormalizer
from pixel_repair import PixelRepairPipeline

import h5py
import numpy as np


class VNIRImageProcessor:
    
    def __init__(self):
        WINDOW_SIZE = 3
        MEAN_TOL = 3
        N_NEIGHBORS = 1
        IMAGE_LABEL = 'VNIR'

        self.repair_pipeline = PixelRepairPipeline(window_size=WINDOW_SIZE, mean_tol=MEAN_TOL, n_neighbors=N_NEIGHBORS)
        self.ref_normalizer = ReflectanceNormalizer()
        self.image_label = IMAGE_LABEL

    
    @staticmethod
    def read_image(path, image_label, transpose=True):
        h5 = h5py.File(path,'r')
        image = h5[image_label]['image']
        if transpose:
            return np.transpose(image, (0, 2, 1))
        else:
            return np.transpose(image, (0, 1, 2))


    def initialize_processor(self, dark_ref_path, light_ref_path):
        dark_ref_image = self.read_image(dark_ref_path, self.image_label)
        light_ref_image = self.read_image(light_ref_path, self.image_label)
        self.ref_normalizer.get_dark_light_offsets(dark_ref_img=dark_ref_image, light_ref_img=light_ref_image)

    
    def process_image(self, image):
        image = self.ref_normalizer.fix_reflectance(image)
        return image
