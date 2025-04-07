import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
import os
import subprocess
from pathlib import Path

# Superpoint keypoint detection
from src.localization_package.SuperPoint.keypoint_detector import get_keypoints

# Utility to check autograd graph
# from torchviz import make_dot

########################### Load NERF Model Config. ###########################

class Localizer():
    def __init__(self, learning_rate=0.01, max_iters=1000, render=False, stream=False):
        """
        learning_rate: learning rate for optimization
        max_iters: maximum number of optimization iterations
        fixed_z: fixed z coordinate (camera height) -- For ground rovers
        save_render: Save intermediate renders to file
        stream: Stream intermediate renders to same file for live update
        """
        # Initialize parameters
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.render_viz = render
        self.stream = stream
        self.device = torch.device('cuda')

    def build_pose(self, x, y, z, roll, pitch, yaw):
        # Create optimizable pose parameters to query NeRF -- Start at small non-zero values to avoid local minima
        pass

    def keypoint_matcher(pose=None, keypoint_threshold=0.005, match_threshold=0.2):
        # Run Superpoint to detect keypoints on nerf render and robot image, then use SuperGlue for keypoint matching 
        superglue_matcher_path = os.path.join(Path.cwd().parent, "SuperGlue", "match_pairs.py")
        input_dir = os.path.join(Path.cwd(), "localizer_images")
        input_pairs = os.path.join(input_dir, "image_pair.txt")
        output_dir = os.path.join(Path.cwd(), "matches")
        subprocess.run(["python3", superglue_matcher_path,
                        "--input_pairs", input_pairs,
                        "--input_dir", input_dir,
                        "--output_dir", output_dir,
                        "--keypoint_threshold", str(keypoint_threshold),
                        "--match_threshold", str(match_threshold),
                        "--viz"
                        ])
        
# Test
Localizer().keypoint_matcher()

# image_path = "1.png"

# Localizer().build_pose(image_path)

