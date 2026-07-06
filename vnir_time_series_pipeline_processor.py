from image_reader_plotter import ImageReaderPlotter
from vnir_image_processor import VNIRImageProcessor
from sift_alignment import SiftAlignment
from ransac_shift_estimation import RansacShiftEstimation

import numpy as np


class VNIRTimeSeriesPipelineProcessor:


    def __init__(self):

        self.parent_folder = '/home/sotgab/sysops/HIPPA/bruising/'
        self.output_path = '/home/sotgab/sysops/HIPPA/GAB/full_preprocessing_pipeline/outputs/VNIR/'
        dark_ref_path = '/home/sotgab/sysops/HIPPA/bruising/references/MICROTEC_test_dark2_251020.h5'
        light_ref_path = '/home/sotgab/sysops/HIPPA/bruising/references/MICROTEC_test_spectralon_251020.h5'
        self.target_channels = 'VNIR'
        self.channels_to_trim = 50
        self.crop = 400

        ### SIFT PARAMS ###

        nfeatures = 300
        nOctaveLayers = 3 # default is 3
        contrastThreshold = 0.02 # default is 0.04, the higher less features are retained
        edgeThreshold = 10 # deafault is 10, the higher more features are retained

        ### RANSAC PARAMS ###

        min_samples=10
        ransac_samples=5
        ransac_threshold=2
        ransac_iterations=100
        min_inlier_ratio=0.25

        self.img_reader_plotter = ImageReaderPlotter()
        self.filenames_dict = self.img_reader_plotter.group_files_by_prefix(self.parent_folder)
        self.img_proc = VNIRImageProcessor()
        self.img_proc.initialize_processor(dark_ref_path=dark_ref_path, light_ref_path=light_ref_path)

        self.sift_align = SiftAlignment(
            nfeatures=nfeatures,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold
        )

        self.ransac_shift = RansacShiftEstimation(
            min_samples=min_samples, 
            ransac_samples=ransac_samples, 
            ransac_threshold=ransac_threshold, 
            ransac_iterations=ransac_iterations, 
            min_inlier_ratio=min_inlier_ratio
        )

    
    def preprocess_image(self, image):
        image = self.img_proc.process_image(image)
        image = self.img_reader_plotter.trim_image_channels(image, self.channels_to_trim)
        image = self.img_reader_plotter.crop_image(image, self.crop)
        return image


    def time_series_alignment_pipeline(self, target_apple, alignment_type='ransac_median', verbose=False):

        filenames = self.filenames_dict[target_apple]

        print('\n\n>>> READING AND PREPROCESSING ALL TIME POINTS IMAGES <<<')

        images = []
        for fn in filenames:
            image = self.img_reader_plotter.read_image(self.parent_folder + fn, self.target_channels)
            images.append(image)
        
        t3_img = self.preprocess_image(images[3])
        t2_img = self.preprocess_image(images[2])
        t1_img = self.preprocess_image(images[1])
        t0_img = self.preprocess_image(images[0])

        print('\n\n>>> SAVING T0 AND T3 PROCESSED IMAGES <<<')
        self.img_reader_plotter.save_image(t0_img, self.output_path + filenames[0], self.target_channels)
        self.img_reader_plotter.save_image(t3_img, self.output_path + filenames[3], self.target_channels)

        print('\n\n>>> STARTING ALIGNMENT OF T2 WITH T3 <<<')

        if alignment_type == 'bands_median':
            t2_t3_shift = self.bands_median_alignment(t3_img, t2_img, verbose)
        elif alignment_type == 'ransac_median':
            t2_t3_shift = self.ransac_median_alignment(t3_img, t2_img, verbose)
        else:
            pass
        
        print('\n\n>>> SHIFTING T2 <<<')
        if t2_t3_shift == 0.0:
            t2_shifted_img = t2_img
        else:
            t2_shifted_img = self.img_reader_plotter.dy_shift_image(t2_img, t2_t3_shift)

        print('\n\n>>> SAVING T2 SHIFTED IMAGE <<<')
        self.img_reader_plotter.save_image(t2_shifted_img, self.output_path + filenames[2], self.target_channels)

        print('\n\n>>> STARTING ALIGNMENT OF T1 WITH T2 <<<')    

        if alignment_type == 'bands_median':
            t1_t2_shift = self.bands_median_alignment(t2_shifted_img, t1_img, verbose)
        elif alignment_type == 'ransac_median':
            t1_t2_shift = self.ransac_median_alignment(t2_shifted_img, t1_img, verbose)
        else:
            pass
        
        print('\n\n>>> SHIFTING T1 <<<')
        if t1_t2_shift == 0.0:
            t1_shifted_img = t1_img
        else:
            t1_shifted_img = self.img_reader_plotter.dy_shift_image(t1_img, t1_t2_shift)

        print('\n\n>>> SAVING T1 SHIFTED IMAGE <<<')
        self.img_reader_plotter.save_image(t1_shifted_img, self.output_path + filenames[1], self.target_channels)


    def bands_median_alignment(self, ref_img, mov_img, verbose=False):
        num_bands = ref_img.shape[2]
        bands_median_list = []
        for test_band in range(num_bands):

            ref_gs_img = ref_img[:, :, test_band]
            mov_gs_img = mov_img[:, :, test_band]

            try:
                pts_ref, pts_mov, good_matches, kp1, kp2 = self.sift_align.detect_and_match_sift(ref_gs_img, mov_gs_img)
                deltas = pts_ref - pts_mov
                if len(deltas) == 0:
                    dy = 0
                dy = np.median(deltas[:, 1])
                if verbose:
                    print(f'>>> Band {test_band}: number of good matches = {len(good_matches)} | estimated shift = {dy}')
                bands_median_list.append(dy)
            except:
                pass
        return np.median(bands_median_list)


    def ransac_median_alignment(self, ref_img, mov_img, verbose=False):
        num_bands = ref_img.shape[2]
        best_shifts_list = []
        for test_band in range(num_bands):

            ref_gs_img = ref_img[:, :, test_band]
            mov_gs_img = mov_img[:, :, test_band]

            try:
                pts_ref, pts_mov, good_matches, kp1, kp2 = self.sift_align.detect_and_match_sift(ref_gs_img, mov_gs_img)
                deltas = pts_ref - pts_mov
                y_deltas = deltas[:, 1]
                best_shift, best_inliers = self.ransac_shift.ransac_shift(y_deltas)
                if len(best_inliers) > 0:
                    if verbose:
                        print(f'>>> Band {test_band}: number of RANSAC inliers = {sum(best_inliers)} | best shift = {best_shift}')
                    best_shifts_list.append(best_shift)
                else:
                    if verbose:
                        print(f'>>> Band {test_band}: RANSAC failed')
            except:
                pass
        
        if len(best_shifts_list) < 3:
            return 0.0
        else:
            return np.median(best_shifts_list)
    