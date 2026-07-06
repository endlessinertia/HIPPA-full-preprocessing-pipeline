import numpy as np

class PixelRepairPipeline:
    
    def __init__(self, window_size, mean_tol, n_neighbors):
        self.window_size = window_size
        self.mean_tol = mean_tol
        self.n_neighbors = n_neighbors
        self.bands_pixel_idx_dict = dict()
    
    @staticmethod
    def find_outliers(values, window_size, mean_tol):
        outliers = np.zeros(len(values), dtype=bool)
        
        for i in range(len(values)):
            start = max(0, i - window_size)
            end = min(len(values), i + window_size + 1)
            neighbors = np.concatenate([values[start:i], values[i+1:end]])
            
            local_mean = np.mean(neighbors)            
            if values[i] > mean_tol * local_mean:
                outliers[i] = True    
        return outliers
    
    def get_broken_pixels(self, ref_img, verbose=False):
        bands = ref_img.shape[2]
        for b in range(bands):
            data = np.mean(ref_img[:,:,b], axis=0)
            outlier_mask = self.find_outliers(data, 
                                              window_size=self.window_size, 
                                              mean_tol=self.mean_tol)
            
            if any(outlier_mask):
                if verbose:
                    print(f'\n## Ref SWIR - Band {b} ##')
                    print("Outliers:", data[outlier_mask])
                    print("Indices:", np.where(outlier_mask)[0])
                self.bands_pixel_idx_dict[b] = np.where(outlier_mask)[0].tolist()


    @staticmethod
    def smooth_image_band_columns(image_band, col_indexes_list, n_neighbors):
        n_col = image_band.shape[1]
        for col_index in col_indexes_list:
            neighbors = []
            for i in range(max(0, col_index - n_neighbors), min(n_col, col_index + n_neighbors + 1)):
                if i != col_index:
                    neighbors.append(i)
            image_band[:, col_index] = np.mean(image_band[:, neighbors], axis=1)
        return image_band
    
    
    def repair_image(self, image):
        # image = image.copy()
        for band in self.bands_pixel_idx_dict:
            image_band = image[:,:,band]
            image[:,:,band] = self.smooth_image_band_columns(image_band, 
                                                             col_indexes_list=self.bands_pixel_idx_dict[band], 
                                                             n_neighbors=self.n_neighbors)        
        return image