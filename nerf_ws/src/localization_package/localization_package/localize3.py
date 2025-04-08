import torch
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import yaml
import os
import subprocess
from pathlib import Path


# Function to render image from NeRF NeRF bassed on camera position specifiled by pose matrix
from nerf_renderer import NeRF_Renderer

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
        self.loss_threshold = 0.001
        self.render_viz = render
        self.stream = stream
        self.device = torch.device('cuda')

    def build_pose_and_render(self, x, y, z, roll, pitch, yaw, image_height, image_width):
        # Create optimizable pose parameters to query NeRF -- Start at small non-zero values to avoid local minima
        pose_params = torch.tensor([x, y, z, pitch, roll, yaw], device='cuda', requires_grad=True)

        # Build Pose Matrix
        # Extract Parameters
        x, y, z, roll, pitch, yaw = pose_params[0], pose_params[1], pose_params[2], pose_params[3], pose_params[4], pose_params[5]

        cos_x, sin_x = torch.cos(pitch), torch.sin(pitch)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        cos_z, sin_z = torch.cos(roll), torch.sin(roll)

        # Rotation matrix for roll
        Rx = torch.tensor([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ], device='cuda', dtype=torch.float32)

        # Rotation matrix for pitch
        Ry = torch.tensor([
            [-cos_y, 0, sin_y],
            [0, -1, 0],
            [-sin_y, 0, cos_y]
        ], device='cuda', dtype=torch.float32)

        # Rotation matrix for yaw
        Rz = torch.tensor([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ], device='cuda', dtype=torch.float32)

        # Combine Rotation Matrix
        R = torch.mm(torch.mm(Rz, Ry), Rx)

        # Create translation vector
        t = torch.zeros((3, 1), device='cuda')
        t[0, 0] = x
        t[1, 0] = y
        t[2, 0] = z

        # Combine rotation matrix and translation vector (top 3x4 part)
        top_rows = torch.cat([R, t], dim=1)
        
        # Add homogeneous (bottom) row
        bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device='cuda')
        # Combine all into pose matrix
        pose_matrix = torch.cat([top_rows, bottom_row], dim=0).unsqueeze(0) # Add batch dim

        # render NeRF image bassed on pose_matrix
        NeRF_Renderer().nerf_image(pose_matrix, image_height, image_width)
        
        return pose_matrix-

    def keypoint_matcher(pose_matrix=None, keypoint_threshold=0.005, match_threshold=0.2):
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
        
    def localize_pose(self):
        
        # Initialize optimizer
        optimizer = torch.optim.SGD([self.pose_params], lr=self.learning_rate, weight_decay=1e-3) # Use Classic gradient descent with L2 Regularization (weight decay)

        if self.render:
            # Create nerf_renders directory to store NeRF Renders
            render_dir = os.path.join(os.getcwd(), "nerf_renders")
            os.makedirs(render_dir, exist_ok=True)

        iter = 0
        # Optimization loop -- Keep optimizing until loss is below threshold or max iterations reached
        while self.loss_value > self.loss_threshold:

            if iter > self.max_iters:
                print("Max iterations Reached")
                break

            loss = ... 
            
            iter += 1




        
        
# Test
robot_image_path = "localizer_images/nerf_size.png"
robot_image = imageio.imread(robot_image_path)

image_height, image_width, _ = robot_image.shape
print(f'-----------------------------{image_height, image_width}------------------------------------------')

Localizer().build_pose_and_render(x=0.1, y=0.1, z=0.1, roll=0.05, pitch=0.05, yaw=0.05, image_height=480, image_width=640)
Localizer().keypoint_matcher()


# Localizer().build_pose(image_path)



