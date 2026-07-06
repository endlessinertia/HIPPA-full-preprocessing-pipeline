import numpy as np

class ReflectanceNormalizer:
    
    def __init__(self):
        self.dark_offset = None
        self.light_offset = None
    
    def get_dark_light_offsets(self, dark_ref_img, light_ref_img):
        self.dark_offset = np.mean(dark_ref_img[:,:,:], axis=0)
        self.light_offset = np.mean(light_ref_img[:,:,:], axis=0)
    
    def fix_reflectance(self, image):
        image = (image - self.dark_offset) / (self.light_offset - self.dark_offset)
        return (image * 255).astype(np.uint8)
