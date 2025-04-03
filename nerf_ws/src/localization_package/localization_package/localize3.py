import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
import os

# NeRF Functions
from nerf_config.libs.nerf.utils import *
from nerf_config.libs.nerf.provider import NeRFDataset
from nerf_config.libs.nerf.network import NeRFNetwork
from nerf_config.config.model_options import ModelOptions

# Superpoint keypoint detection
from src.localization_package.SuperPoint.keypoint_detector import get_keypoints

# Utiity to check autograd graph
from torchviz import make_dot

########################### Load NERF Model Config. ###########################

def load_config(file_path):
    """Load YAML configuration file with environment variable expansion."""
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)

        def expand_vars(item):
            if isinstance(item, str):
                return os.path.expandvars(item)
            elif isinstance(item, dict):
                return {key: expand_vars(value) for key, value in item.items()}
            elif isinstance(item, list):
                return [expand_vars(elem) for elem in item]
            else:
                return item

        config = expand_vars(config)
        return config

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration file: {e}")

########################### END Load NERF Model Config. ###########################

class Localizer():
    def __init__(self, learning_rate=0.01, max_iters=1000, batch_size=2048, kernel_size=5, dilate_iter=5, render=False, stream=False):
        """
        render_fn: function taking rays (origins, directions) and returning a dict with key 'image'
        get_rays_fn: function taking a 4x4 camera pose matrix and returning dict with 'rays_o' and 'rays_d'
        learning_rate: learning rate for optimization
        max_iters: maximum number of iterations for optimization if not converged
        kernal_size: size of the kernal used for dilation
        dilate_iter: number of iterations for dilation
        batch_size: number of pixels (from feature regions) used per iteration
        fixed_z: fixed z coordinate (camera height) -- For ground rovers
        save_render: Save intermediate renders to file
        stream: Stream intermediate renders to same file for live update
        """
        # Initialize parameters
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.dilate_iter = dilate_iter
        self.render_viz = render
        self.stream = stream
        self.device = torch.device('cuda')

    def build_pose(self, image_path, x=0.01, y=0.01, z=0.01, yaw=0.05): # OR call NeRF_Render
        # Detect Image Keypoints
        keypoints, scores, descriptors = get_keypoints(image_path, weights_path='../SuperPoint/weights/superpoint_v6_from_tf.pth')
        if keypoints.shape == 0:
            print("No Features Found in Image")
            return None
        print(keypoints)

        keypoints=keypoints.astype(np.uint8)

        # Create optimizable (trainable) pose parameters -- with small non-zero values to avoid local minima





# ## Test
image_path = "1.png"

Localizer().build_pose(image_path)
# weights_path = '../SuperPoint/weights/superpoint_v6_from_tf.pth'
# keypoints, scores, descriptors = get_keypoints(image_path, weights_path)