import numpy as np
import matplotlib.pyplot as plt

def get_rgb_bands():

    vnir_calib_round = np.load('/home/sotgab/sysops/HIPPA/bruising/references/vis_hippa_wl_calib_rounded.npy')

    # Target wavelengths (nm)
    TARGET_R = 660
    TARGET_G = 550
    TARGET_B = 470

    def find_band(wavelengths, target_wl):
        return np.argmin(np.abs(wavelengths - target_wl))
    
    rgb_bands = [
        find_band(vnir_calib_round, TARGET_R),
        find_band(vnir_calib_round, TARGET_G),
        find_band(vnir_calib_round, TARGET_B)
        ]
    print(f'RGB bands indexes are: {rgb_bands}')
    print(f'Which corresponds to {vnir_calib_round[rgb_bands]} nanometers')

    return rgb_bands


def make_pseudo_rgb(hs_image, rgb_bands, normalize_channels=False):

    rgb_image = np.stack([
        hs_image[:, :, rgb_bands[0]],
        hs_image[:, :, rgb_bands[1]],
        hs_image[:, :, rgb_bands[2]],
    ], axis=-1)

    if normalize_channels:
        # Normalize each channel independently to [0, 1]
        rgb_image = rgb_image.astype(np.float32)
        for c in range(3):
            ch = rgb_image[:, :, c]
            rgb_image[:, :, c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

    return rgb_image

def save_rgb_image(rgb_image, path):
    plt.imsave(path, rgb_image)
