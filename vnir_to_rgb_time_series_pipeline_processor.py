from image_reader_plotter import ImageReaderPlotter
from vnir_to_rgb import *

import numpy as np


class VNIRtoRGBTimeSeriesPipeline:


    def __init__(self):

        self.parent_folder = '/home/sotgab/sysops/HIPPA/bruising/'
        self.output_path = '/home/sotgab/sysops/HIPPA/bruising/processed/RGB/'
        dark_ref_path = '/home/sotgab/sysops/HIPPA/bruising/references/MICROTEC_test_dark2_251020.h5'
        light_ref_path = '/home/sotgab/sysops/HIPPA/bruising/references/MICROTEC_test_spectralon_251020.h5'

        self.target_channels = 'VNIR'
        self.crop = 600
        self.normalize_channel = True

        self.img_reader_plotter = ImageReaderPlotter()
        self.filenames_dict = self.img_reader_plotter.group_files_by_prefix(self.parent_folder)

    
    def preprocess_image(self, image):
        image = self.img_reader_plotter.crop_image(image, self.crop)
        image = self.img_reader_plotter.mirror_image(image)
        return image


    def time_series_to_rgb_pipeline(self, target_apple, alignment_type='ransac_median', verbose=False):

        filenames = self.filenames_dict[target_apple]
        rgb_bands = get_rgb_bands()

        print('\n\n>>> READING AND PREPROCESSING ALL TIME POINTS IMAGES <<<')

        images = []
        for fn in filenames:
            image = self.img_reader_plotter.read_image(self.parent_folder + fn, self.target_channels)
            images.append(image)
        
        t3_img = self.preprocess_image(images[3])
        t2_img = self.preprocess_image(images[2])
        t1_img = self.preprocess_image(images[1])
        t0_img = self.preprocess_image(images[0])

        print('\n\n>>> CONVERTING ALL TIME POINTS IMAGES INTO RGB <<<')

        t3_rgb_img = make_pseudo_rgb(t3_img, rgb_bands, normalize_channels=self.normalize_channel)
        t2_rgb_img = make_pseudo_rgb(t2_img, rgb_bands, normalize_channels=self.normalize_channel)
        t1_rgb_img = make_pseudo_rgb(t1_img, rgb_bands, normalize_channels=self.normalize_channel)
        t0_rgb_img = make_pseudo_rgb(t0_img, rgb_bands, normalize_channels=self.normalize_channel)

        print('\n\n>>> SAVING ALL TIME POINTS RGB IMAGES INTO PNG <<<')

        save_rgb_image(t3_rgb_img, self.output_path + filenames[3][:-3] + '.png')
        save_rgb_image(t2_rgb_img, self.output_path + filenames[2][:-3] + '.png')
        save_rgb_image(t1_rgb_img, self.output_path + filenames[1][:-3] + '.png')
        save_rgb_image(t0_rgb_img, self.output_path + filenames[0][:-3] + '.png')

    