import random
import numbers
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.stats
import numpy as np
import torch
import copy
from collections.abc import Sequence, Mapping
from scipy.spatial.transform import Rotation as R
import time
from scipy.ndimage import gaussian_filter,zoom
from pointcept.utils.registry import Registry

TRANSFORMS = Registry("transforms")

# Rendering Equivalence Augmentations
@TRANSFORMS.register_module()
class GaussianNoiseInjection(object):
    """
    Inject noise to Gaussian positions based on MCMC paper.
    Low-opacity Gaussians get more noise (exploration).
    High-opacity Gaussians get less noise (converged).
    """
    def __init__(self, noise_lr=1e-4, k=100, t=0.005, p=0.5):
        """
        Args:
            noise_lr: noise learning rate (scaled down from paper's 5e5)
            k: sharpness of sigmoid transition
            t: opacity threshold for transition
            p: probability of applying augmentation
        """
        self.noise_lr = noise_lr  # Much smaller than training since we're in data loading
        self.k = k
        self.t = t
        self.p = p
    
    def compute_covariance_from_scale_quat(self, scales, quats):
        """Compute 3D covariance matrices from scales and quaternions."""
        n = len(scales)
        
        # Convert quaternions (w,x,y,z) to rotation matrices
        # scipy expects (x,y,z,w) format
        quats_xyzw = np.concatenate([quats[:, 1:], quats[:, :1]], axis=1)
        rotations = R.from_quat(quats_xyzw).as_matrix()  # [N, 3, 3]
        
        # Create scale matrices S = diag(scales)
        S = np.zeros((n, 3, 3))
        S[:, 0, 0] = scales[:, 0]
        S[:, 1, 1] = scales[:, 1]
        S[:, 2, 2] = scales[:, 2]
        
        # Covariance = R @ S @ S^T @ R^T
        SS = S @ np.transpose(S, (0, 2, 1))  # S @ S^T
        cov = rotations @ SS @ np.transpose(rotations, (0, 2, 1))
        
        return cov
    
    def __call__(self, data_dict):
        if np.random.rand() > self.p:
            return data_dict
            
        if "coord" in data_dict and "opacity" in data_dict and "scale" in data_dict and "quat" in data_dict:
            coords = data_dict["coord"]
            opacities = data_dict["opacity"]
            scales = data_dict["scale"]
            quats = data_dict["quat"]
            
            # Compute covariance matrices
            covs = self.compute_covariance_from_scale_quat(scales, quats)
            
            # Opacity-dependent noise weight using sigmoid
            # High opacity -> sigmoid(negative) -> ~0 (less noise)
            # Low opacity -> sigmoid(positive) -> ~1 (more noise)
            sigmoid_input = -self.k * (self.t - opacities)
            sigmoid_input = np.clip(sigmoid_input, -20, 20)  # Prevent overflow
            opacity_weight = 1.0 / (1.0 + np.exp(-sigmoid_input))
            opacity_weight = opacity_weight.reshape(-1, 1)
            
            # Generate noise
            eta = np.random.randn(*coords.shape)
            
            # Apply covariance-weighted noise
            # noise = lr * opacity_weight * cov @ eta
            noise = np.zeros_like(coords)
            for i in range(len(coords)):
                noise[i] = self.noise_lr * opacity_weight[i] * (covs[i] @ eta[i])
            
            data_dict["coord"] = coords + noise
            
        return data_dict


@TRANSFORMS.register_module()
class GaussianRelocation(object):
    """
    More accurate relocation following MCMC paper formulas.
    """
    def __init__(self, min_opacity=0.006, relocation_rate=0.2, p=0.3):
        self.min_opacity = min_opacity
        self.relocation_rate = relocation_rate
        self.p = p
        
        # Precompute binomial coefficients
        self.binoms = self._compute_binomials(51)
    
    def _compute_binomials(self, n_max):
        """Precompute binomial coefficients."""
        import math
        binoms = np.zeros((n_max, n_max))
        for n in range(n_max):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)
        return binoms
    
    def _compute_new_opacity(self, old_opacity, n_clones):
        """Exact formula from MCMC paper."""
        return 1.0 - np.power(1.0 - old_opacity, 1.0 / n_clones)
    
    def _compute_new_scale(self, old_scale, old_opacity, new_opacity, n_clones):
        """Simplified version of MCMC scale adjustment."""
        # This is a simplified approximation
        # Full formula requires the complex binomial sum
        scale_factor = np.sqrt(old_opacity / (new_opacity * n_clones))
        return old_scale * scale_factor
    
    def __call__(self, data_dict):
        if np.random.rand() > self.p:
            return data_dict
            
        if all(k in data_dict for k in ["opacity", "coord", "scale"]):
            opacities = data_dict["opacity"]
            coords = data_dict["coord"]
            scales = data_dict["scale"]
            
            # Identify dead and alive Gaussians
            dead_mask = opacities < self.min_opacity
            alive_mask = ~dead_mask
            
            n_dead = dead_mask.sum()
            n_alive = alive_mask.sum()
            
            if n_dead > 0 and n_alive > 0:
                n_relocate = min(int(n_dead * self.relocation_rate), n_alive)
                
                if n_relocate > 0:
                    dead_indices = np.where(dead_mask)[0]
                    selected_dead = np.random.choice(dead_indices, n_relocate, replace=False)
                    
                    # Sample targets weighted by opacity
                    alive_indices = np.where(alive_mask)[0]
                    alive_opacities = opacities[alive_mask]
                    probs = alive_opacities / alive_opacities.sum()
                    selected_alive = np.random.choice(alive_indices, n_relocate, p=probs, replace=True)
                    
                    # Count how many Gaussians are relocated to each target
                    unique_targets, counts = np.unique(selected_alive, return_counts=True)
                    
                    # Apply MCMC formulas for each unique target
                    for target, n_clones in zip(unique_targets, counts):
                        # Find which dead Gaussians go to this target
                        going_to_target = selected_dead[selected_alive == target]
                        
                        # Compute new opacity (exact formula)
                        old_opacity = opacities[target]
                        new_opacity = self._compute_new_opacity(old_opacity, n_clones + 1)
                        
                        # Update opacities
                        opacities[target] = new_opacity
                        opacities[going_to_target] = new_opacity
                        
                        # Update positions with small noise
                        coords[going_to_target] = coords[target] + np.random.randn(len(going_to_target), 3) * 0.001
                        
                        # Update scales (simplified)
                        old_scale = scales[target]
                        new_scale = self._compute_new_scale(old_scale, old_opacity, new_opacity, n_clones + 1)
                        scales[target] = new_scale
                        scales[going_to_target] = new_scale
                        
                        # Copy quaternions
                        if "quat" in data_dict:
                            data_dict["quat"][going_to_target] = data_dict["quat"][target]
                    
                    data_dict["coord"] = coords
                    data_dict["opacity"] = opacities
                    data_dict["scale"] = scales
        
        return data_dict
    
# Immature Manifold Augmentations
@TRANSFORMS.register_module()
class GaussianScaleBlur(object):
    """
    Simulate blur by increasing Gaussian scales.
    Based on RAIN-GS: larger scales produce blurrier renders.
    """
    def __init__(self, blur_factor_range=[1.0, 3.0], size_dependent=True, p=0.3,
                max_scale=1.0, eps=1e-6):
        """
        Args:
            blur_factor_range: range of scale multiplication factors
            size_dependent: if True, apply more blur to small (high-freq) Gaussians
            p: probability of applying augmentation
        """
        self.blur_factor_range = blur_factor_range
        self.size_dependent = size_dependent
        self.p = p
        self.max_scale = max_scale
        self.eps = eps
    
    def __call__(self, data_dict):
        if np.random.rand() > self.p:
            return data_dict
            
        if "scale" in data_dict:
            scales = data_dict["scale"].copy()
            
            # Random blur factor
            blur_factor = np.random.uniform(*self.blur_factor_range)
            
            if self.size_dependent:
                scale_magnitude = np.mean(scales, axis=1, keepdims=True)

                finite_mask = np.isfinite(scale_magnitude)
                finite_vals = scale_magnitude[finite_mask]
                if finite_vals.size == 0:
                    data_dict["scale"] = scales
                    return data_dict

                median_scale = np.median(finite_vals[finite_vals > 0]) if np.any(finite_vals > 0) else 0.0
                denom = max(median_scale, self.eps)

                # clamp exponent to avoid under/overflow
                with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                    exponent = -scale_magnitude / denom
                    exponent = np.clip(exponent, -20.0, 0.0)  # exp(-20) ~ 2e-9
                    weight = np.exp(exponent)

                # any invalid weights -> 0
                invalid_w = ~np.isfinite(weight)
                if np.any(invalid_w):
                    weight[invalid_w] = 0.0

                # Also ensure weight within [0,1]
                weight = np.clip(weight, 0.0, 1.0)
                effective_blur = 1.0 + (blur_factor - 1.0) * weight

                # invalid effective_blur -> 1.0 (no-op)
                invalid_eb = ~np.isfinite(effective_blur)
                if np.any(invalid_eb):
                    effective_blur[invalid_eb] = 1.0

                scales = scales * effective_blur
            else:
                # Uniform blur
                scales = scales * blur_factor
            
            # Ensure scales don't become too large (based on data statistics)
            scales = np.clip(scales, self.eps, self.max_scale)  # Max scale around 0.73 in data
            
            data_dict["scale"] = scales
        
        return data_dict


@TRANSFORMS.register_module()
class GaussianFrequencyNoise(object):
    """
    Add noise based on frequency (scale).
    Small scales (high-freq) get more noise, large scales (low-freq) get less.
    Based on RAIN-GS frequency analysis.
    """
    def __init__(self, high_freq_noise=0.005, low_freq_ratio=0.05, p=0.5):
        """
        Args:
            high_freq_noise: noise level for high-frequency (small) Gaussians
            low_freq_ratio: ratio of noise for low-frequency Gaussians
            p: probability of applying augmentation
        """
        self.high_freq_noise = high_freq_noise
        self.low_freq_ratio = low_freq_ratio
        self.p = p
    
    def __call__(self, data_dict):
        if np.random.rand() > self.p:
            return data_dict
            
        if "coord" in data_dict and "scale" in data_dict:
            coords = data_dict["coord"]
            scales = data_dict["scale"]
            
            # Compute scale magnitudes
            scale_magnitude = np.mean(scales, axis=1)
            
            # Use median to separate high/low frequency
            median_scale = np.median(scale_magnitude)
            
            # Create frequency-dependent noise weights
            # Small scales (< median) -> weight close to 1 (more noise)
            # Large scales (> median) -> weight close to low_freq_ratio (less noise)
            weights = np.ones_like(scale_magnitude)
            high_freq_mask = scale_magnitude < median_scale
            low_freq_mask = ~high_freq_mask
            
            # Smooth transition using sigmoid
            relative_scale = scale_magnitude / median_scale
            x = np.clip(2.0 * (1.0 - relative_scale), -10, 10)
            weights = 1.0 / (1.0 + np.exp(-x))
            weights = self.low_freq_ratio + (1.0 - self.low_freq_ratio) * weights
            weights = weights.reshape(-1, 1)
            
            # Generate and apply noise
            noise = np.random.randn(*coords.shape) * self.high_freq_noise * weights
            data_dict["coord"] = coords + noise
            
            # Optionally add noise to scales too (smaller effect)
            if np.random.rand() < 0.5:
                scale_noise = np.random.randn(*scales.shape) * 0.001 * weights
                data_dict["scale"] = scales * (1.0 + scale_noise)
        
        return data_dict

# SSL attribute transforms
@TRANSFORMS.register_module()
class CollectContrast(object):
    def __init__(self, keys_prefix, offset_keys_dict=None, **kwargs):
        """
        e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        # keys=("coord", "grid_coord", "segment"),
        # feat_keys=("color", "normal"),
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord") # {'offset': 'coord'}
            # use offset to get the shape of the data
        self.keys = keys_prefix
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs



    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            for key_i in data_dict.keys():
                if key_i.startswith(key):
                    data[key_i] = data_dict[key_i]
                    # print(key_i, data_dict[key_i].shape)
            # data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]]) 
        for name, keys in self.kwargs.items():
            # print("name", name)
            # print("keys", keys)
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1) # featre concat

        return data


# efficent version
@TRANSFORMS.register_module()
class GSGaussianBlurVoxelOpc(object):
    def __init__(self, p=0.5, sigma=[0.1,2,0], extra_keys=None):
        self.p = p
        self.sigma = sigma
        self.extra_keys = extra_keys
        

    def __call__(self, data_dict):
        # efficient for 3D point cloud color blur
        t0 = time.time()
        if np.random.rand() < self.p:

            assert 'grid_coord' in data_dict.keys(), f"grid_coord is required for GSGaussianBlur, but only {data_dict.keys()} is provided"
            coord = data_dict["coord"] # already go through grid sampler and crop operation
            grid_coord = data_dict["grid_coord"] # is uniformed!
            opacity = data_dict["opacity"]
            random_sigma = np.random.uniform(self.sigma[0], self.sigma[1])

            # we only blur the color with opacity > 0.2
            blur_mask = opacity > 0.5
            blur_mask = blur_mask.ravel()
            grid_coord_masked = grid_coord[blur_mask]  # Use ravel() instead of reshape(-1)

            grid_coord_max = grid_coord.max(axis=0)
            grid_coord_min = grid_coord.min(axis=0)
            grid_size = grid_coord_max - grid_coord_min + 1
            # Convert coordinates to grid indices
            grid_indices = (grid_coord_masked - grid_coord_min).astype(int)

            # Create compact color and weight grids using broadcasting
            color_grid = np.zeros((*grid_size, 3), dtype=np.float32)
            weight_grid = np.zeros((*grid_size, 1), dtype=np.float32)
            color_index = [0,1,2]
            weight_index = [3]
            grid_index = [0,1,2,3]
            # for extra key
            if 'opacity' in self.extra_keys:
                opacity_grid = np.zeros((*grid_size, 1), dtype=np.float32)
                opacity_index = np.array([0]) + len(grid_index)
                grid_index.extend(opacity_index.tolist())
            if 'scale' in self.extra_keys:
                scale_grid = np.zeros((*grid_size, 3), dtype=np.float32)
                scale_index = np.array([0,1,2]) + len(grid_index)
                grid_index.extend(scale_index.tolist())
            if 'quat' in self.extra_keys:
                quat_grid = np.zeros((*grid_size, 4), dtype=np.float32)
                quat_index = np.array([0,1,2,3]) + len(grid_index)
                grid_index.extend(quat_index.tolist())
                # do achieve lerp + normalize, 
            if "normal" in self.extra_keys:
                normal_grid = np.zeros((*grid_size, 3), dtype=np.float32)
                normal_index = np.array([0,1,2]) + len(grid_index)
                grid_index.extend(normal_index.tolist())

            # Vectorized assignment using advanced indexing
            color_grid[tuple(grid_indices.T)] = data_dict["color"][blur_mask]
            weight_grid[tuple(grid_indices.T)] = 1  # Use 3D array for weights

            # print("color_grid", color_grid.shape, "sigma", random_sigma)
            # print("weight_grid", weight_grid.shape)

            feature_grid = np.concatenate([color_grid, weight_grid], axis=-1)

            if 'opacity' in self.extra_keys:
                opacity_grid[tuple(grid_indices.T)] = data_dict["opacity"][blur_mask]
                feature_grid = np.concatenate([feature_grid, opacity_grid], axis=-1)
            if 'scale' in self.extra_keys:
                scale_grid[tuple(grid_indices.T)] = data_dict["scale"][blur_mask]
                feature_grid = np.concatenate([feature_grid, scale_grid], axis=-1)
            if 'quat' in self.extra_keys:
                quat_grid[tuple(grid_indices.T)] = data_dict["quat"][blur_mask]
                feature_grid = np.concatenate([feature_grid, quat_grid], axis=-1)
            if "normal" in self.extra_keys:
                normal_grid[tuple(grid_indices.T)] = data_dict["normal"][blur_mask]
                feature_grid = np.concatenate([feature_grid, normal_grid], axis=-1)
            # we can concate all feature together and do the blur


            truncate = 2.0
            # print("color_grid", color_grid.shape, "sigma", random_sigma)
            # Use float32 for better memory usage and faster computation
            blur_feature_grid = gaussian_filter(feature_grid, sigma=random_sigma,truncate=truncate, axes=(0, 1, 2))
            blur_color = blur_feature_grid[..., color_index]
            blur_weights = blur_feature_grid[..., weight_index] + 1e-7  # Prevent division by zero

            # Get updated colors through advanced indexing
            result_colors = data_dict["color"].copy()
            result_colors[blur_mask] = blur_color[tuple(grid_indices.T)] / blur_weights[tuple(grid_indices.T)]

            data_dict["color"] = result_colors

            if 'opacity' in self.extra_keys:
                result_opacity = data_dict["opacity"].copy()
                result_opacity[blur_mask] = blur_feature_grid[tuple(grid_indices.T)][..., opacity_index] / blur_weights[tuple(grid_indices.T)]
                # print("result_opacity", result_opacity.shape)
                data_dict["opacity"] = result_opacity
            if 'scale' in self.extra_keys:
                result_scale = data_dict["scale"].copy()
                result_scale[blur_mask] = blur_feature_grid[tuple(grid_indices.T)][..., scale_index] / blur_weights[tuple(grid_indices.T)]
                # print("result_scale", result_scale.shape)
                data_dict["scale"] = result_scale
            if 'quat' in self.extra_keys:
                result_quat = data_dict["quat"].copy()
                result_quat[blur_mask] = blur_feature_grid[tuple(grid_indices.T)][..., quat_index] / blur_weights[tuple(grid_indices.T)]
                # renormalization
                result_quat = result_quat / np.linalg.norm(result_quat, axis=1, keepdims=True)
                # print("result_quat", result_quat.shape)
                data_dict["quat"] = result_quat
            if "normal" in self.extra_keys:
                result_normal = data_dict["normal"].copy()
                result_normal[blur_mask] = blur_feature_grid[tuple(grid_indices.T)][..., normal_index] / blur_weights[tuple(grid_indices.T)]
                # print("result_normal", result_normal.shape)
                data_dict["normal"] = result_normal


        return data_dict

@TRANSFORMS.register_module()
class RandomColorSolarize(object):
    def __init__(self, p=0.2, threshold=128):
        self.p = p
        self.threshold = threshold

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:   
            masked_color = data_dict["color"]
            mask = data_dict["color"] < self.threshold
            # change to add and substract
            mask_sign = np.ones_like(mask) 
            mask_sign[mask] = -1
            mask_add_on = np.zeros_like(masked_color)
            mask_add_on[mask] = 255
            masked_color = masked_color * mask_sign + mask_add_on
        return data_dict
    

@TRANSFORMS.register_module()
class SphereCropRandomMaxPoints(object):
    """
    reduce the point cloud to a fixed maximum number of points
    """
    def __init__(self, random_scale=[0.5,1.0], point_max=80000):
        self.point_max = point_max
        self.random_scale = random_scale
        self.point_max = point_max


    def __call__(self, data_dict):

        assert "coord" in data_dict.keys()
        # pts_num = data_dict["coord"].shape[0]
        point_max = int(np.random.uniform(self.random_scale[0], self.random_scale[1]) * self.point_max)
        if data_dict["coord"].shape[0] > point_max:
            center = data_dict["coord"][
                np.random.randint(data_dict["coord"].shape[0])
            ]

            idx_crop = np.argsort(np.sum(np.square(data_dict["coord"] - center), 1))[
                :point_max
            ]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx_crop]
            if "origin_coord" in data_dict.keys():
                data_dict["origin_coord"] = data_dict["origin_coord"][idx_crop]
            if "grid_coord" in data_dict.keys():
                data_dict["grid_coord"] = data_dict["grid_coord"][idx_crop]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx_crop]
            if "quat" in data_dict.keys():
                data_dict["quat"] = data_dict["quat"][idx_crop]
            if "scale" in data_dict.keys():
                data_dict["scale"] = data_dict["scale"][idx_crop]
            if "opacity" in data_dict.keys():
                data_dict["opacity"] = data_dict["opacity"][idx_crop]
            if "sh" in data_dict.keys():
                data_dict["sh"] = data_dict["sh"][idx_crop]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx_crop]
            if "lang_feat" in data_dict.keys():
                data_dict["lang_feat"] = data_dict["lang_feat"][idx_crop]
            if "lang_feat_64" in data_dict.keys():
                data_dict["lang_feat_64"] = data_dict["lang_feat_64"][idx_crop]
            if "valid_feat_mask" in data_dict.keys():
                data_dict["valid_feat_mask"] = data_dict["valid_feat_mask"][idx_crop]


            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][idx_crop]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx_crop]
            if "displacement" in data_dict.keys():
                data_dict["displacement"] = data_dict["displacement"][idx_crop]
            if "strength" in data_dict.keys():
                data_dict["strength"] = data_dict["strength"][idx_crop]
        return data_dict



@TRANSFORMS.register_module()
class ContrastiveViewsGenerator_SSL(object):
    def __init__(
        self,
        view_keys=("coord", "color", "normal", "origin_coord"),
        # basic_trans_cfg=None, # already in basic_trans_cfg before
        global_base_transform=None,
        local_base_transform=None,
        global_transform0=None,
        global_transform1=None,
        local_transform=None,
        local_crop_num=4,
    ):
        self.view_keys = view_keys
        # self.basic_trans = Compose(basic_trans_cfg)
        # follow dinov2, we have two global transform and one local transform
        self.global_base_transform = Compose(global_base_transform)
        self.local_base_transform = Compose(local_base_transform)
        self.global_transform0 = Compose(global_transform0)
        self.global_transform1 = Compose(global_transform1)
        self.local_transform = Compose(local_transform)
        self.local_crop_num = local_crop_num


    def __call__(self, data_dict):
        # print("data_dict", data_dict.keys())
        global_base_dict = dict()
        local_base_dict = dict()
        # we want lang feat to be similar in both global view
        for key in self.view_keys:
            global_base_dict[key] = data_dict[key].copy()
            local_base_dict[key] = data_dict[key].copy()
        global_base_dict = self.global_base_transform(global_base_dict)
        global_crop_1_dict = {key: global_base_dict[key].copy() for key in self.view_keys}
        global_crop_2_dict = {key: global_base_dict[key].copy() for key in self.view_keys}

        global_crop_1_dict = self.global_transform0(global_crop_1_dict)
        global_crop_2_dict = self.global_transform1(global_crop_2_dict)

        local_crop_dict_list = []
        local_base_dict = self.local_base_transform(local_base_dict)
        for i in range(self.local_crop_num):
            local_crop_dict = {key: local_base_dict[key].copy() for key in self.view_keys}
            local_crop_dict = self.local_transform(local_crop_dict)
            local_crop_dict_list.append(local_crop_dict)

        # collect results
        for key, value in global_crop_1_dict.items():
            data_dict["global_crop0_" + key] = value
        for key, value in global_crop_2_dict.items():
            data_dict["global_crop1_" + key] = value

        for i, local_crop_dict in enumerate(local_crop_dict_list):
            for key, value in local_crop_dict.items():
                data_dict["local_crop{}_".format(i) + key] = value

        return data_dict
    


@TRANSFORMS.register_module()
class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        """
        e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        # keys=("coord", "grid_coord", "segment"),
        # feat_keys=("color", "normal"),
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")  # {'offset': 'coord'}
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            if key in data_dict.keys():
                data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor(
                [data_dict[value].shape[0]]
            )  # record the shape of the data
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            # for key_i in keys:
            #     print(key_i, data_dict[key_i].shape)
            data[name] = torch.cat(
                [data_dict[key].float() for key in keys], dim=1
            )  # feat_keys concat
        return data


@TRANSFORMS.register_module()
class Copy(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict(coord="origin_coord", segment="origin_segment")
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            if key in data_dict.keys():
                if isinstance(data_dict[key], np.ndarray):
                    data_dict[value] = data_dict[key].copy()
                elif isinstance(data_dict[key], torch.Tensor):
                    data_dict[value] = data_dict[key].clone().detach()
                else:
                    data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict


@TRANSFORMS.register_module()
class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            if data.dtype == np.float64:
                return torch.from_numpy(data).float()  # float64 → float32
            else:
                return torch.from_numpy(data)
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@TRANSFORMS.register_module()
class Add(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict()
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            data_dict[key] = value
        return data_dict


@TRANSFORMS.register_module()
class NormalizeColor(object):
    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"] / 127.5 - 1
        return data_dict


@TRANSFORMS.register_module()
class NormalizeCoord(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            # modified from pointnet2
            centroid = np.mean(data_dict["coord"], axis=0)
            data_dict["coord"] -= centroid
            m = np.max(np.sqrt(np.sum(data_dict["coord"] ** 2, axis=1)))
            data_dict["coord"] = data_dict["coord"] / m
        if "scale" in data_dict.keys():
            data_dict["scale"] = data_dict["scale"] / m
        if "pc_coord" in data_dict.keys():
            data_dict["pc_coord"] -= centroid
            data_dict["pc_coord"] = data_dict["pc_coord"] / m
        return data_dict


@TRANSFORMS.register_module()
class PositiveShift(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord_min = np.min(data_dict["coord"], 0)
            data_dict["coord"] -= coord_min

            if "pc_coord" in data_dict.keys():
                # we apply the same shift to pc_coord
                data_dict["pc_coord"] -= coord_min
        return data_dict


@TRANSFORMS.register_module()
class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            x_min, y_min, z_min = data_dict["coord"].min(axis=0)
            x_max, y_max, _ = data_dict["coord"].max(axis=0)
            if self.apply_z:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
            else:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
            data_dict["coord"] -= shift

            if "pc_coord" in data_dict.keys():
                # we apply the same shift to pc_coord
                data_dict["pc_coord"] -= shift

        return data_dict


@TRANSFORMS.register_module()
class RandomShift(object):
    def __init__(self, shift=((-0.2, 0.2), (-0.2, 0.2), (0, 0))):
        self.shift = shift

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            shift_x = np.random.uniform(self.shift[0][0], self.shift[0][1])
            shift_y = np.random.uniform(self.shift[1][0], self.shift[1][1])
            shift_z = np.random.uniform(self.shift[2][0], self.shift[2][1])
            data_dict["coord"] += [shift_x, shift_y, shift_z]
            if "pc_coord" in data_dict.keys():
                # we apply the same shift to pc_coord
                data_dict["pc_coord"] += [shift_x, shift_y, shift_z]
        return data_dict


@TRANSFORMS.register_module()
class PointClip(object):
    def __init__(self, point_cloud_range=(-80, -80, -3, 80, 80, 1)):
        self.point_cloud_range = point_cloud_range

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            data_dict["coord"] = np.clip(
                data_dict["coord"],
                a_min=self.point_cloud_range[:3],
                a_max=self.point_cloud_range[3:],
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["coord"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx = np.unique(np.append(idx, data_dict["sampled_index"]))

                # remaps the original indices of the important points to their new positions in the subsampled data
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx])[0]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx]
            if "strength" in data_dict.keys():
                data_dict["strength"] = data_dict["strength"][idx]
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][idx]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx]
            if "superpoint" in data_dict.keys():
                data_dict["superpoint"] = data_dict["superpoint"][idx]

            if "quat" in data_dict.keys():
                data_dict["quat"] = data_dict["quat"][idx]
            if "scale" in data_dict.keys():
                data_dict["scale"] = data_dict["scale"][idx]
            if "opacity" in data_dict.keys():
                data_dict["opacity"] = data_dict["opacity"][idx]
            if "lang_feat" in data_dict.keys():
                data_dict["lang_feat"] = data_dict["lang_feat"][idx]
            if "dino_feat" in data_dict.keys():
                data_dict["dino_feat"] = data_dict["dino_feat"][idx]
            if "pe_feat" in data_dict.keys():
                data_dict["pe_feat"] = data_dict["pe_feat"][idx]
            if "valid_feat_mask" in data_dict.keys():
                data_dict["valid_feat_mask"] = data_dict["valid_feat_mask"][idx]
        return data_dict


@TRANSFORMS.register_module()
class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError

        # Rotate coordinates (and pc_coord if available)
        if "coord" in data_dict:
            if self.center is None:
                # compute center of bounding box
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] = (data_dict["coord"] - center) @ rot_t.T + center
            if "pc_coord" in data_dict:
                data_dict["pc_coord"] = (
                    data_dict["pc_coord"] - center
                ) @ rot_t.T + center

        # Rotate quaternions.
        if "quat" in data_dict:
            # 3DGS stores quaternions in wxyz format.
            # SciPy expects quaternions in xyzw order.
            # Convert from wxyz to xyzw by rolling left by 1.
            quat_wxyz = data_dict["quat"]
            quat_xyzw = np.roll(quat_wxyz, shift=-1, axis=1)
            input_quat = R.from_quat(quat_xyzw)
            rot = R.from_matrix(rot_t)
            # Apply the rotation: left-multiply with the global rotation.
            new_quat_xyzw = (rot * input_quat).as_quat()
            # Convert back from xyzw to wxyz by rolling right by 1.
            new_quat_wxyz = np.roll(new_quat_xyzw, shift=1, axis=1)
            data_dict["quat"] = new_quat_wxyz

        # Rotate normals if present
        if "normal" in data_dict:
            data_dict["normal"] = data_dict["normal"] @ rot_t.T

        return data_dict


@TRANSFORMS.register_module()
class RandomRotateTargetAngle(object):
    def __init__(
        self, angle=(1 / 2, 1, 3 / 2), center=None, axis="z", always_apply=False, p=0.75
    ):
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.choice(self.angle) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        dtype = data_dict["coord"].dtype
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t)).astype(dtype)
            data_dict["coord"] += center
            if "pc_coord" in data_dict.keys():
                data_dict["pc_coord"] -= center
                data_dict["pc_coord"] = np.dot(
                    data_dict["pc_coord"], np.transpose(rot_t)
                )
                data_dict["pc_coord"] += center
        if "quat" in data_dict:
            # 3DGS stores quaternions in wxyz format.
            # SciPy expects quaternions in xyzw order.
            # Convert from wxyz to xyzw by rolling left by 1.
            quat_wxyz = data_dict["quat"]
            quat_xyzw = np.roll(quat_wxyz, shift=-1, axis=1)
            input_quat = R.from_quat(quat_xyzw)
            rot = R.from_matrix(rot_t)
            # Apply the rotation: left-multiply with the global rotation.
            new_quat_xyzw = (rot * input_quat).as_quat()
            # Convert back from xyzw to wxyz by rolling right by 1.
            new_quat_wxyz = np.roll(new_quat_xyzw, shift=1, axis=1).astype(dtype)
            data_dict["quat"] = new_quat_wxyz
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


@TRANSFORMS.register_module()
class RandomScale(object):
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            scale = np.random.uniform(
                self.scale[0], self.scale[1], 3 if self.anisotropic else 1
            )
            data_dict["coord"] *= scale
            if "pc_coord" in data_dict.keys():
                data_dict["pc_coord"] *= scale
            if "scale" in data_dict.keys():
                data_dict["scale"] *= scale
        return data_dict


@TRANSFORMS.register_module()
class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        R_reflect = np.eye(3)
        flipped = False

        if np.random.rand() < self.p:
            # Flip along x-axis.
            reflect_x = np.diag([-1, 1, 1])
            R_reflect = reflect_x @ R_reflect
            flipped = True
            if "coord" in data_dict:
                data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
            if "pc_coord" in data_dict:
                data_dict["pc_coord"][:, 0] = -data_dict["pc_coord"][:, 0]
            if "normal" in data_dict:
                data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]

        if np.random.rand() < self.p:
            # Flip along y-axis.
            reflect_y = np.diag([1, -1, 1])
            R_reflect = reflect_y @ R_reflect
            flipped = True
            if "coord" in data_dict:
                data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
            if "pc_coord" in data_dict:
                data_dict["pc_coord"][:, 1] = -data_dict["pc_coord"][:, 1]
            if "normal" in data_dict:
                data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]

        if flipped and "quat" in data_dict:
            # 3DGS stores quaternions in wxyz order.
            # SciPy expects xyzw, so convert by rolling left.
            quat_wxyz = data_dict["quat"]
            quat_xyzw = np.roll(quat_wxyz, shift=-1, axis=1)
            current_rot = R.from_quat(quat_xyzw).as_matrix()
            new_rot = R_reflect @ current_rot @ R_reflect
            new_quat_xyzw = R.from_matrix(new_rot).as_quat()

            # Convert back to wxyz by rolling right.
            new_quat_wxyz = np.roll(new_quat_xyzw, shift=1, axis=1)
            data_dict["quat"] = new_quat_wxyz

        return data_dict


@TRANSFORMS.register_module()
class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.clip(
                self.sigma * np.random.randn(data_dict["coord"].shape[0], 3),
                -self.clip,
                self.clip,
            )
            data_dict["coord"] += jitter
        return data_dict


@TRANSFORMS.register_module()
class ClipGaussianJitter(object):
    def __init__(self, scalar=0.02, store_jitter=False):
        self.scalar = scalar
        self.mean = np.mean(3)
        self.cov = np.identity(3)
        self.quantile = 1.96
        self.store_jitter = store_jitter

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.random.multivariate_normal(
                self.mean, self.cov, data_dict["coord"].shape[0]
            )
            jitter = self.scalar * np.clip(jitter / 1.96, -1, 1)
            data_dict["coord"] += jitter
            if self.store_jitter:
                data_dict["jitter"] = jitter
        return data_dict


@TRANSFORMS.register_module()
class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            lo = np.min(data_dict["color"], 0, keepdims=True)
            hi = np.max(data_dict["color"], 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (data_dict["color"][:, :3] - lo) * scale
            blend_factor = (
                np.random.rand() if self.blend_factor is None else self.blend_factor
            )
            data_dict["color"][:, :3] = (1 - blend_factor) * data_dict["color"][
                :, :3
            ] + blend_factor * contrast_feat
        return data_dict


@TRANSFORMS.register_module()
class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data_dict["color"][:, :3] = np.clip(tr + data_dict["color"][:, :3], 0, 255)
        return data_dict


@TRANSFORMS.register_module()
class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            noise = np.random.randn(data_dict["color"].shape[0], 3)
            noise *= self.std * 255
            data_dict["color"][:, :3] = np.clip(
                noise + data_dict["color"][:, :3], 0, 255
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomColorGrayScale(object):
    def __init__(self, p):
        self.p = p

    @staticmethod
    def rgb_to_grayscale(color, num_output_channels=1):
        if color.shape[-1] < 3:
            raise TypeError(
                "Input color should have at least 3 dimensions, but found {}".format(
                    color.shape[-1]
                )
            )

        if num_output_channels not in (1, 3):
            raise ValueError("num_output_channels should be either 1 or 3")

        r, g, b = color[..., 0], color[..., 1], color[..., 2]
        gray = (0.2989 * r + 0.587 * g + 0.114 * b).astype(color.dtype)
        gray = np.expand_dims(gray, axis=-1)

        if num_output_channels == 3:
            gray = np.broadcast_to(gray, color.shape)

        return gray

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            data_dict["color"] = self.rgb_to_grayscale(data_dict["color"], 3)
        return data_dict


@TRANSFORMS.register_module()
class RandomColorJitter(object):
    """
    Random Color Jitter for 3D point cloud (refer torchvision)
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.95):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.p = p

    @staticmethod
    def _check_input(
        value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def blend(color1, color2, ratio):
        ratio = float(ratio)
        bound = 255.0
        return (
            (ratio * color1 + (1.0 - ratio) * color2)
            .clip(0, bound)
            .astype(color1.dtype)
        )

    @staticmethod
    def rgb2hsv(rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb, axis=-1)
        minc = np.min(rgb, axis=-1)
        eqc = maxc == minc
        cr = maxc - minc
        s = cr / (np.ones_like(maxc) * eqc + maxc * (1 - eqc))
        cr_divisor = np.ones_like(maxc) * eqc + cr * (1 - eqc)
        rc = (maxc - r) / cr_divisor
        gc = (maxc - g) / cr_divisor
        bc = (maxc - b) / cr_divisor

        hr = (maxc == r) * (bc - gc)
        hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
        hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
        h = hr + hg + hb
        h = (h / 6.0 + 1.0) % 1.0
        return np.stack((h, s, maxc), axis=-1)

    @staticmethod
    def hsv2rgb(hsv):
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = np.floor(h * 6.0)
        f = (h * 6.0) - i
        i = i.astype(np.int32)

        p = np.clip((v * (1.0 - s)), 0.0, 1.0)
        q = np.clip((v * (1.0 - s * f)), 0.0, 1.0)
        t = np.clip((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
        i = i % 6
        mask = np.expand_dims(i, axis=-1) == np.arange(6)

        a1 = np.stack((v, q, p, p, t, v), axis=-1)
        a2 = np.stack((t, v, v, q, p, p), axis=-1)
        a3 = np.stack((p, p, t, v, v, q), axis=-1)
        a4 = np.stack((a1, a2, a3), axis=-1)

        return np.einsum("...na, ...nab -> ...nb", mask.astype(hsv.dtype), a4)

    def adjust_brightness(self, color, brightness_factor):
        if brightness_factor < 0:
            raise ValueError(
                "brightness_factor ({}) is not non-negative.".format(brightness_factor)
            )

        return self.blend(color, np.zeros_like(color), brightness_factor)

    def adjust_contrast(self, color, contrast_factor):
        if contrast_factor < 0:
            raise ValueError(
                "contrast_factor ({}) is not non-negative.".format(contrast_factor)
            )
        mean = np.mean(RandomColorGrayScale.rgb_to_grayscale(color))
        return self.blend(color, mean, contrast_factor)

    def adjust_saturation(self, color, saturation_factor):
        if saturation_factor < 0:
            raise ValueError(
                "saturation_factor ({}) is not non-negative.".format(saturation_factor)
            )
        gray = RandomColorGrayScale.rgb_to_grayscale(color)
        return self.blend(color, gray, saturation_factor)

    def adjust_hue(self, color, hue_factor):
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError(
                "hue_factor ({}) is not in [-0.5, 0.5].".format(hue_factor)
            )
        orig_dtype = color.dtype
        hsv = self.rgb2hsv(color / 255.0)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h = (h + hue_factor) % 1.0
        hsv = np.stack((h, s, v), axis=-1)
        color_hue_adj = (self.hsv2rgb(hsv) * 255.0).astype(orig_dtype)
        return color_hue_adj

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        b = (
            None
            if brightness is None
            else np.random.uniform(brightness[0], brightness[1])
        )
        c = None if contrast is None else np.random.uniform(contrast[0], contrast[1])
        s = (
            None
            if saturation is None
            else np.random.uniform(saturation[0], saturation[1])
        )
        h = None if hue is None else np.random.uniform(hue[0], hue[1])
        return fn_idx, b, c, s, h

    def __call__(self, data_dict):
        (
            fn_idx,
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
        ) = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if (
                fn_id == 0
                and brightness_factor is not None
                and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_brightness(
                    data_dict["color"], brightness_factor
                )
            elif (
                fn_id == 1 and contrast_factor is not None and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_contrast(
                    data_dict["color"], contrast_factor
                )
            elif (
                fn_id == 2
                and saturation_factor is not None
                and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_saturation(
                    data_dict["color"], saturation_factor
                )
            elif fn_id == 3 and hue_factor is not None and np.random.rand() < self.p:
                data_dict["color"] = self.adjust_hue(data_dict["color"], hue_factor)
        return data_dict


@TRANSFORMS.register_module()
class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype("float")
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select(
            [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc
        )
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype("uint8")
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype("uint8")

    def __init__(self, hue_max=0.5, saturation_max=0.2):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            # Assume color[:, :3] is rgb
            hsv = HueSaturationTranslation.rgb_to_hsv(data_dict["color"][:, :3])
            hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
            sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
            hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
            hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
            data_dict["color"][:, :3] = np.clip(
                HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomColorDrop(object):
    def __init__(self, p=0.2, color_augment=0.0):
        self.p = p
        self.color_augment = color_augment

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            data_dict["color"] *= self.color_augment
        return data_dict

    def __repr__(self):
        return "RandomColorDrop(color_augment: {}, p: {})".format(
            self.color_augment, self.p
        )


@TRANSFORMS.register_module()
class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    data_dict["coord"] = self.elastic_distortion(
                        data_dict["coord"], granularity, magnitude
                    )
        return data_dict


@TRANSFORMS.register_module()
class GridSample(object):
    #  select one representative point per cell
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        keys=("coord", "color", "normal", "segment"),
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
        importance_sample_key=None,
        apply_to_pc=True,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement
        self.importance_sample_key = importance_sample_key
        self.apply_to_pc = apply_to_pc

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)  # normalize to 0
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

        if "pc_coord" in data_dict and self.apply_to_pc:
            pc_coord = data_dict["pc_coord"]

            # ── integer grid local to pc_coord ──────────────────────────────────────
            pc_scaled_coord = pc_coord / np.asarray(self.grid_size)
            pc_grid_coord = np.floor(pc_scaled_coord).astype(int)
            pc_grid_coord -= pc_grid_coord.min(0)

            # ── hashing & sorting ───────────────────────────────────────────────────
            pc_key = self.hash(pc_grid_coord)
            pc_idx_sort = np.argsort(pc_key, kind="stable")
            pc_key_sorted = pc_key[pc_idx_sort]

            # start indices of each grid cell in the sorted list
            first_idx = np.nonzero(
                np.concatenate(([True], pc_key_sorted[1:] != pc_key_sorted[:-1]))
            )[0]

            pc_segment = data_dict.get("pc_segment", None)
            chosen_idx = []

            for i, start in enumerate(first_idx):
                end = first_idx[i + 1] if i + 1 < len(first_idx) else len(pc_idx_sort)
                cell_idx = pc_idx_sort[start:end]  # len ≥ 1

                if pc_segment is not None:
                    valid = cell_idx[pc_segment[cell_idx] != -1]
                    chosen_idx.append(valid[0] if len(valid) else cell_idx[0])
                else:
                    chosen_idx.append(cell_idx[0])

            chosen_idx = np.asarray(chosen_idx, dtype=np.int64)

            # ── subsample the point cloud and its segment labels ────────────────────
            data_dict["pc_coord"] = data_dict["pc_coord"][chosen_idx]
            if "pc_segment" in data_dict:
                data_dict["pc_segment"] = data_dict["pc_segment"][chosen_idx]
            if "pc_instance" in data_dict:
                data_dict["pc_instance"] = data_dict["pc_instance"][chosen_idx]

        if self.mode == "train":  # train mode
            if self.importance_sample_key is None:
                idx_select = (
                    np.cumsum(np.insert(count, 0, 0)[0:-1])
                    + np.random.randint(0, count.max(), count.size) % count
                )
                idx_unique = idx_sort[idx_select]
                # start_idx = np.cumsum(np.insert(count, 0, 0)[:-1])
                # random_offsets = (
                #     np.random.randint(0, count.max(), count.size) % count
                # )
                # idx_select = start_idx + random_offsets
                # if "segment" in data_dict:
                #     # prioritize voxels containing labeled points; fallback to random selection otherwise
                #     segment_sorted = data_dict["segment"][idx_sort]
                #     valid_mask = segment_sorted != -1
                #     if np.any(valid_mask):
                #         cell_ids = np.repeat(np.arange(count.size), count)
                #         valid_indices = np.flatnonzero(valid_mask)
                #         valid_cells = cell_ids[valid_indices]
                #         perm = np.random.permutation(valid_indices.size)
                #         shuffled_valid_indices = valid_indices[perm]
                #         shuffled_valid_cells = valid_cells[perm]
                #         unique_cells, first_occ = np.unique(
                #             shuffled_valid_cells, return_index=True
                #         )
                #         idx_select[unique_cells] = shuffled_valid_indices[first_occ]
                # idx_unique = idx_sort[idx_select]
            else:
                idx_unique = np.array(
                    self.importance_sample(idx_sort, count, data_dict)
                )
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
            for key in self.keys:
                if key in data_dict.keys():  # data may not have normal
                    data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                if self.return_inverse:
                    data_dict["inverse"] = np.zeros_like(inverse)
                    data_dict["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_dict["displacement"] = displacement[idx_part]
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                data_part_list.append(data_part)
            return data_part_list  # covers a subset of points (one “slice” of each voxel cell)
        else:
            raise NotImplementedError

    def importance_sample(self, idx_sort, count, data_dict):
        if isinstance(self.importance_sample_key, tuple):
            for subkey in self.importance_sample_key:
                if subkey not in data_dict and not (
                    "scale" in subkey and "scale" in data_dict
                ):
                    raise ValueError(
                        "Importance sample key {} not found in data_dict".format(subkey)
                    )
            importance_sample = None
            for subkey in self.importance_sample_key:
                if "scale" in subkey:
                    if subkey.split("_")[1] == "max":
                        # keep the max scale of 3 channels
                        importance_attribute = np.max(data_dict["scale"], axis=-1)
                    elif subkey.split("_")[1] == "prod":
                        importance_attribute = np.prod(data_dict["scale"], axis=-1)
                    elif subkey.split("_")[1] == "min":
                        importance_attribute = np.min(data_dict["scale"], axis=-1)
                    else:
                        raise ValueError(
                            "Importance sample key {} not found in data_dict".format(
                                subkey
                            )
                        )
                else:
                    importance_attribute = data_dict[subkey]
                if importance_sample is None:
                    importance_sample = importance_attribute
                else:
                    importance_sample = importance_sample * importance_attribute
        else:
            if (
                self.importance_sample_key not in data_dict
                or self.importance_sample_key not in self.keys
            ):
                raise ValueError(
                    "Importance sample key {} not found in data_dict".format(
                        self.importance_sample_key
                    )
                )
            importance_sample = data_dict[self.importance_sample_key]

        grid_splits = np.cumsum(count[:-1])
        grid_indices_list = np.split(idx_sort, grid_splits)

        return [grid[importance_sample[grid].argmax()] for grid in grid_indices_list]

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


@TRANSFORMS.register_module()
class SphereCrop(object):
    """
    reduce the point cloud to a fixed maximum number of points
    """

    def __init__(self, point_max=80000, sample_rate=None, mode="random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "all"]
        self.mode = mode

    def __call__(self, data_dict):
        point_max = (
            int(self.sample_rate * data_dict["coord"].shape[0])
            if self.sample_rate is not None
            else self.point_max
        )

        assert "coord" in data_dict.keys()
        if self.mode == "all":
            # TODO: Optimize
            if "index" not in data_dict.keys():
                data_dict["index"] = np.arange(data_dict["coord"].shape[0])
            data_part_list = []
            # coord_list, color_list, dist2_list, idx_list, offset_list = [], [], [], [], []
            if data_dict["coord"].shape[0] > point_max:
                coord_p, idx_uni = (
                    np.random.rand(data_dict["coord"].shape[0]) * 1e-3,
                    np.array([]),
                )
                while idx_uni.size != data_dict["index"].shape[0]:
                    init_idx = np.argmin(coord_p)
                    dist2 = np.sum(
                        np.power(data_dict["coord"] - data_dict["coord"][init_idx], 2),
                        1,
                    )
                    idx_crop = np.argsort(dist2)[:point_max]

                    data_crop_dict = dict()
                    print("data_dict.keys()", data_dict.keys())

                    if "coord" in data_dict.keys():
                        data_crop_dict["coord"] = data_dict["coord"][idx_crop]
                    if "grid_coord" in data_dict.keys():
                        data_crop_dict["grid_coord"] = data_dict["grid_coord"][idx_crop]
                    if "normal" in data_dict.keys():
                        data_crop_dict["normal"] = data_dict["normal"][idx_crop]
                    if "color" in data_dict.keys():
                        data_crop_dict["color"] = data_dict["color"][idx_crop]
                    if "opacity" in data_dict.keys():
                        # print("crop opacity", len(data_dict["opacity"]),'to',len(data_dict["opacity"][idx_crop]))
                        data_crop_dict["opacity"] = data_dict["opacity"][idx_crop]
                    if "quat" in data_dict.keys():
                        data_crop_dict["quat"] = data_dict["quat"][idx_crop]
                    if "lang_feat" in data_dict.keys():
                        data_crop_dict["lang_feat"] = data_dict["lang_feat"][idx_crop]
                    if "dino_feat" in data_dict.keys():
                        data_dict["dino_feat"] = data_dict["dino_feat"][idx_crop]
                    if "pe_feat" in data_dict.keys():
                        data_dict["pe_feat"] = data_dict["pe_feat"][idx_crop]
                    if "valid_feat_mask" in data_dict.keys():
                        data_crop_dict["valid_feat_mask"] = data_dict[
                            "valid_feat_mask"
                        ][idx_crop]

                    if "scale" in data_dict.keys():
                        data_crop_dict["scale"] = data_dict["scale"][idx_crop]
                    if "superpoint" in data_dict.keys():
                        data_dict["superpoint"] = data_dict["superpoint"][idx_crop]
                    if "displacement" in data_dict.keys():
                        data_crop_dict["displacement"] = data_dict["displacement"][
                            idx_crop
                        ]
                    if "strength" in data_dict.keys():
                        data_crop_dict["strength"] = data_dict["strength"][idx_crop]
                    data_crop_dict["weight"] = dist2[idx_crop]
                    data_crop_dict["index"] = data_dict["index"][idx_crop]
                    data_part_list.append(data_crop_dict)

                    delta = np.square(
                        1 - data_crop_dict["weight"] / np.max(data_crop_dict["weight"])
                    )
                    coord_p[idx_crop] += delta
                    idx_uni = np.unique(
                        np.concatenate((idx_uni, data_crop_dict["index"]))
                    )
            else:
                data_crop_dict = data_dict.copy()
                data_crop_dict["weight"] = np.zeros(data_dict["coord"].shape[0])
                data_crop_dict["index"] = data_dict["index"]
                data_part_list.append(data_crop_dict)
            return data_part_list
        # mode is "random" or "center"
        elif data_dict["coord"].shape[0] > point_max:
            if self.mode == "random":
                center = data_dict["coord"][
                    np.random.randint(data_dict["coord"].shape[0])
                ]
            elif self.mode == "center":
                center = data_dict["coord"][data_dict["coord"].shape[0] // 2]
            else:
                raise NotImplementedError
            idx_crop = np.argsort(np.sum(np.square(data_dict["coord"] - center), 1))[
                :point_max
            ] # here choose the closest points to the center point
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx_crop]
            if "origin_coord" in data_dict.keys():
                data_dict["origin_coord"] = data_dict["origin_coord"][idx_crop]
            if "grid_coord" in data_dict.keys():
                data_dict["grid_coord"] = data_dict["grid_coord"][idx_crop]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx_crop]
            if "quat" in data_dict.keys():
                data_dict["quat"] = data_dict["quat"][idx_crop]
            if "scale" in data_dict.keys():
                data_dict["scale"] = data_dict["scale"][idx_crop]
            if "opacity" in data_dict.keys():
                data_dict["opacity"] = data_dict["opacity"][idx_crop]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx_crop]
            if "lang_feat" in data_dict.keys():
                data_dict["lang_feat"] = data_dict["lang_feat"][idx_crop]
            if "dino_feat" in data_dict.keys():
                data_dict["dino_feat"] = data_dict["dino_feat"][idx_crop]
            if "pe_feat" in data_dict.keys():
                data_dict["pe_feat"] = data_dict["pe_feat"][idx_crop]
            if "valid_feat_mask" in data_dict.keys():
                data_dict["valid_feat_mask"] = data_dict["valid_feat_mask"][idx_crop]

            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][idx_crop]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx_crop]
            if "superpoint" in data_dict.keys():
                data_dict["superpoint"] = data_dict["superpoint"][idx_crop]
            if "displacement" in data_dict.keys():
                data_dict["displacement"] = data_dict["displacement"][idx_crop]
            if "strength" in data_dict.keys():
                data_dict["strength"] = data_dict["strength"][idx_crop]
        return data_dict


@TRANSFORMS.register_module()
class ShufflePoint(object):
    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        shuffle_index = np.arange(data_dict["coord"].shape[0])
        np.random.shuffle(shuffle_index)
        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"][shuffle_index]
        if "grid_coord" in data_dict.keys():
            data_dict["grid_coord"] = data_dict["grid_coord"][shuffle_index]
        if "displacement" in data_dict.keys():
            data_dict["displacement"] = data_dict["displacement"][shuffle_index]
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"][shuffle_index]
        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"][shuffle_index]
        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"][shuffle_index]
        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"][shuffle_index]
        return data_dict


@TRANSFORMS.register_module()
class CropBoundary(object):
    def __call__(self, data_dict):
        assert "segment" in data_dict
        segment = data_dict["segment"].flatten()
        mask = (segment != 0) * (segment != 1)
        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"][mask]
        if "grid_coord" in data_dict.keys():
            data_dict["grid_coord"] = data_dict["grid_coord"][mask]
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"][mask]
        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"][mask]
        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"][mask]
        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"][mask]
        return data_dict


@TRANSFORMS.register_module()
class ContrastiveViewsGenerator(object):
    def __init__(
        self,
        view_keys=("coord", "color", "normal", "origin_coord"),
        view_trans_cfg=None,
    ):
        self.view_keys = view_keys
        self.view_trans = Compose(view_trans_cfg)

    def __call__(self, data_dict):
        view1_dict = dict()
        view2_dict = dict()
        for key in self.view_keys:
            view1_dict[key] = data_dict[key].copy()
            view2_dict[key] = data_dict[key].copy()
        view1_dict = self.view_trans(view1_dict)
        view2_dict = self.view_trans(view2_dict)
        for key, value in view1_dict.items():
            data_dict["view1_" + key] = value
        for key, value in view2_dict.items():
            data_dict["view2_" + key] = value
        return data_dict


@TRANSFORMS.register_module()
class InstanceParser(object):
    def __init__(self, segment_ignore_index=(-1, 0, 1), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        mask = ~np.in1d(segment, self.segment_ignore_index)
        # mapping ignored instance to ignore index
        instance[~mask] = self.instance_ignore_index
        # reorder left instance
        unique, inverse = np.unique(instance[mask], return_inverse=True)
        instance_num = len(unique)
        instance[mask] = inverse
        # init instance information
        centroid = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        bbox = np.ones((instance_num, 8)) * self.instance_ignore_index
        vacancy = [
            index for index in self.segment_ignore_index if index >= 0
        ]  # vacate class index

        for instance_id in range(instance_num):
            mask_ = instance == instance_id
            coord_ = coord[mask_]
            bbox_min = coord_.min(0)
            bbox_max = coord_.max(0)
            bbox_centroid = coord_.mean(0)
            bbox_center = (bbox_max + bbox_min) / 2
            bbox_size = bbox_max - bbox_min
            bbox_theta = np.zeros(1, dtype=coord_.dtype)
            bbox_class = np.array([segment[mask_][0]], dtype=coord_.dtype)
            # shift class index to fill vacate class index caused by segment ignore index
            bbox_class -= np.greater(bbox_class, vacancy).sum()

            centroid[mask_] = bbox_centroid
            bbox[instance_id] = np.concatenate(
                [bbox_center, bbox_size, bbox_theta, bbox_class]
            )  # 3 + 3 + 1 + 1 = 8
        data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["bbox"] = bbox
        return data_dict


class Compose(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict


@TRANSFORMS.register_module()
class GSGaussianBlurVoxelGPU(object):
    def __init__(self, p=0.5, sigma=[0.1,2,0]):
        self.p = p
        self.sigma = sigma
        

    def __call__(self, data_dict):
        # efficient for 3D point cloud color blur
        t0 = time.time()
        # get view_i from data_dict
        for key_i in data_dict.keys():
            if "view_" in key_i:
                view_prefix = key_i.split("_")[0]
            else:
                view_prefix = None
        
        if np.random.rand() < self.p:
            assert 'grid_coord' in data_dict.keys(), f"grid_coord is required for GSGaussianBlur, but only {data_dict.keys()} is provided"
            grid_coord = data_dict["grid_coord"] # is uniformed!
            # assert grid_coord in gpu tensor of torch
            assert isinstance(grid_coord, torch.Tensor)
            assert grid_coord.is_cuda

            x_grid_num = grid_coord[:, 0].max() - grid_coord[:, 0].min() + 1
            y_grid_num = grid_coord[:, 1].max() - grid_coord[:, 1].min() + 1
            z_grid_num = grid_coord[:, 2].max() - grid_coord[:, 2].min() + 1

            color_grid = torch.zeros((x_grid_num, y_grid_num, z_grid_num, 3)).cuda() # using 1 as default color
            weighted_grid = torch.zeros((x_grid_num, y_grid_num, z_grid_num, 3)).cuda() # using 0 as default shift

            color_grid[grid_coord[:, 0], grid_coord[:, 1], grid_coord[:, 2]] = data_dict["color"] # 10% 0.01 grid
            # in weighted_grid we set the weight to 1 if nonzero
            weighted_grid[grid_coord[:, 0], grid_coord[:, 1], grid_coord[:, 2]] = 1
            # for zero color grid, we fill in with nearest color

            random_sigma = np.random.uniform(self.sigma[0], self.sigma[1])
            radius = round(4.0 * random_sigma)
            # replace gaussian_filter with torch.nn.functional.conv3d
            kernel_3d = torch.ones(1, 1, radius*2+1, radius*2+1, radius*2+1).cuda()
            kernel_3d = kernel_3d / kernel_3d.sum()
            # color_grid to B, C, H, W, D
            color_grid = color_grid.permute(3, 0, 1, 2).unsqueeze(1)
            weighted_grid = weighted_grid.permute(3, 0, 1, 2).unsqueeze(1)
            print("kernel_3d", kernel_3d.shape, "color_grid", color_grid.shape, "weighted_grid", weighted_grid.shape)
            color_grid = torch.nn.functional.conv3d(input=color_grid, weight=kernel_3d,stride=1,padding=radius*2 ).squeeze(1).permute(1, 2, 3, 0)
            weighted_grid = torch.nn.functional.conv3d(input=weighted_grid, weight=kernel_3d,stride=1,padding= radius*2).squeeze(1).permute(1, 2, 3, 0)
            # renormalize the color grid based on weighted grid
            
            blur_color = color_grid[grid_coord[:, 0], grid_coord[:, 1], grid_coord[:, 2]]
            blur_weighted = weighted_grid[grid_coord[:, 0], grid_coord[:, 1], grid_coord[:, 2]]
            blur_color = blur_color / (blur_weighted + 1e-7)
            ### print color difference 
            print("color diff", torch.mean(torch.abs(data_dict["color"] - blur_color).mean(axis=1)), "sigma", random_sigma)
            data_dict["color"] = blur_color
        print("blur time", time.time()-t0)

        return data_dict
