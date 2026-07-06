import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift as nd_shift


class SiftAlignment:

    def __init__(self, 
                nfeatures,
                nOctaveLayers,
                contrastThreshold,
                edgeThreshold):
        
        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold
        )

    
    def detect_and_match_sift(self, ref, mov, ratio_thresh=0.75):
        """
        Detect SIFT keypoints in ref and mov, match them with Lowe's ratio test.
        Returns matched keypoints as (pts_ref, pts_mov) arrays of shape (N, 2).
        """

        kp1, des1 = self.sift.detectAndCompute(ref, None)
        kp2, des2 = self.sift.detectAndCompute(mov, None)

        # BFMatcher with L2 norm (appropriate for SIFT), k=2 for ratio test
        bf = cv2.BFMatcher(cv2.NORM_L2)
        raw_matches = bf.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good_matches = []
        for m, n in raw_matches:
            if m.distance / n.distance < ratio_thresh:
                good_matches.append(m)

        # print(f"  Keypoints: {len(kp1)} / {len(kp2)} | "
        #     f"Raw matches: {len(raw_matches)} | Good matches: {len(good_matches)}")

        pts_ref = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts_mov = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        return pts_ref, pts_mov, good_matches, kp1, kp2
    

    @staticmethod
    def estimate_translation(pts_ref, pts_mov):
        """
        Simple robust translation estimate: median of per-point displacements.
        """
        deltas = pts_ref - pts_mov  # shape (N, 2): (dx, dy) per match
        if len(deltas) == 0:
            return 0, 0
        dx = np.median(deltas[:, 0])
        dy = np.median(deltas[:, 1])
        return dx, dy
    

    @staticmethod
    def estimate_translation_stats(pts_ref, pts_mov, outlier_z_thresh=2.5):
        deltas = pts_ref - pts_mov # shape (N, 2)
        y_deltas = deltas[:, 1]

        mean    = np.mean(y_deltas)
        std     = np.std(y_deltas)
        median  = np.median(y_deltas)
        mad     = np.median(np.abs(y_deltas - median)) # median absolute deviation

        # Modified Z-score outlier detection (robust to non-normality)
        modified_z = 0.6745 * (y_deltas - median) / (mad + 1e-10)
        outlier_mask = np.abs(modified_z) > outlier_z_thresh

        return {
            "y_deltas_mean":            float(mean),
            "y_deltas_std":             float(std),
            "y_deltas_median":          float(median),
            "y_deltas_mad":             float(mad),
            "y_deltas_n_outliers":      int(outlier_mask.sum()),
        }
    

    @staticmethod
    def draw_matches_side_by_side(ref, mov, kp1, kp2, good_matches):
        match_img = cv2.drawMatches(
            ref, kp1, mov, kp2, good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        plt.figure(figsize=(15, 10))
        plt.imshow(match_img, cmap='gray')
        plt.tight_layout()
        plt.show()


    @staticmethod
    def shift_image(img, shift):
        dx, dy = shift
        return nd_shift(img, shift=(dy, dx), mode='wrap')


    def align_with_sift(self, images, images_names):
        """
        Align all images to images[0] using SIFT keypoint matching.
        """
        ref = images[0]
        aligned = [ref]
        transforms = [None]

        for mov, img_name in zip(images[1:], images_names[1:]):
            print(f"\n> Aligning {img_name} with {images_names[0]}:")
            pts_ref, pts_mov, good_matches, kp1, kp2 = self.detect_and_match_sift(ref, mov)
            # self.draw_matches_side_by_side(ref, mov, kp1, kp2, good_matches)

            dx, dy = self.estimate_translation(pts_ref, pts_mov)
            print(f"    Median translation: dx={dx:.2f}, dy={dy:.2f}")
            aligned_img = nd_shift(mov, shift=(dy, dx), mode='wrap')
            transforms.append((dx, dy))

            aligned.append(aligned_img)

        return aligned, transforms
        