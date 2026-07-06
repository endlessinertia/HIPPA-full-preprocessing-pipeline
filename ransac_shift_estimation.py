import numpy as np


class RansacShiftEstimation:

    def __init__(self, min_samples=10, ransac_samples=5, ransac_threshold=2, ransac_iterations=100, min_inlier_ratio=0.5, robust_threshold=2, robust_iterations=100):
        """
        Args:
            min_samples:       Minimum number of samples to attmept the RANSAC estimation.  
            ransac_samples:    Number of random samples to fit each candidate model.
            ransac_threshold:  Max distance from the RANSAC to be considered an inlier.
            ransac_iterations: Number of RANSAC iterations.
            min_inlier_ratio:  Minimum fraction of inliers required to accept a model.
            robust_threshold:  Max distance from the robust estimator to be included as inlier.
            robust_iterations: Number of robust estimator iterations. 
        """
        self.min_samples = min_samples
        self.ransac_samples = ransac_samples
        self.ransac_threshold = ransac_threshold
        self.ransac_iterations = ransac_iterations
        self.min_inlier_ratio = min_inlier_ratio

        self.robust_threshold = robust_threshold
        self.robust_iterations = robust_iterations


    def ransac_shift(self, values, fallback='none'):
        """
        Estimate a robust 1D shift form values list using RANSAC.
        Returns:
            best_shift:   Robust shift estimate.
            inlier_mask:  Boolean mask indicating inliers.
        """
        values = np.asarray(values, dtype=float)
        n = len(values)
        if n < self.min_samples:
            return 0, []

        best_shift = None
        best_inliers = np.zeros(n, dtype=bool)
        best_inlier_count = 0

        for _ in range(self.ransac_iterations):
            # 1. Randomly sample a minimal subset
            sample_idx = np.random.choice(n, size=self.ransac_samples, replace=False)
            candidate_shift = np.median(values[sample_idx])

            # 2. Find all inliers consistent with this candidate
            residuals = np.abs(values - candidate_shift)
            inliers = residuals < self.ransac_threshold

            # 3. Keep the model with the most inliers
            inlier_count = np.sum(inliers)
            if inlier_count > best_inlier_count and inlier_count >= self.min_inlier_ratio * n:
                best_inlier_count = inlier_count
                best_inliers = inliers
                # Refit on all inliers for a better estimate
                best_shift = np.median(values[inliers])

        # Fallback strategy
        if best_shift is None:
            if fallback == 'median':
                best_shift = np.median(values)
                best_inliers = np.ones(n, dtype=bool)
            elif fallback == 'none':
                return 0, []
            else:
                return 0, []

        return best_shift, best_inliers


    def weighted_robust_shift(self, all_lists):
        """
        Robust shift estimation across multiple lists using
        inverse-variance weighting on per-list robust estimates.
        """
        estimates, weights = [], []

        for lst in all_lists:
            arr = np.asarray(lst, dtype=float)

            # Robust per-list estimate (median or RANSAC)
            shift, inliers = self.ransac_shift(arr)

            # Weight = inverse variance of inliers (more consistent = higher weight)
            inlier_vals = arr[inliers]
            if len(inlier_vals) < 2:
                continue
            variance = np.var(inlier_vals)
            weight = len(inlier_vals) / (variance + 1e-9)  # also reward more inliers

            estimates.append(shift)
            weights.append(weight)

        estimates = np.array(estimates)
        weights = np.array(weights)
        weights /= weights.sum()  # normalize

        # Final robust aggregation across list-level estimates as weighted sum
        weighted_mean = np.sum(weights * estimates)

        return weighted_mean