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
        self.loss_threshold = 0.09
        self.render = render
        self.stream = stream
        self.device = torch.device('cuda')

    def build_pose_matrix(self, pose_parameters):
        # Build Pose Matrix from pose_parameters
        # Extract Parameters
        x, y, z, roll, pitch, yaw = pose_parameters[0], pose_parameters[1], pose_parameters[2], pose_parameters[3], pose_parameters[4], pose_parameters[5]

        cos_x, sin_x = torch.cos(pitch), torch.sin(pitch)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        cos_z, sin_z = torch.cos(roll), torch.sin(roll)

        # Rotation matrix for roll
        Rx = torch.tensor([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ], device=self.device, dtype=torch.float32)

        # Rotation matrix for pitch
        Ry = torch.tensor([
            [-cos_y, 0, sin_y],
            [0, -1, 0],
            [-sin_y, 0, cos_y]
        ], device=self.device, dtype=torch.float32)

        # Rotation matrix for yaw
        Rz = torch.tensor([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)

        # Combine Rotation Matrix
        R = torch.mm(torch.mm(Rz, Ry), Rx)

        # Create translation vector
        t = torch.zeros((3, 1), device=self.device)
        t[0, 0] = x
        t[1, 0] = y
        t[2, 0] = z

        # Combine rotation matrix and translation vector (top 3x4 part)
        top_rows = torch.cat([R, t], dim=1)
        
        # Add homogeneous (bottom) row
        bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device)
        # Combine all into pose matrix
        pose_matrix = torch.cat([top_rows, bottom_row], dim=0).unsqueeze(0) # Add batch dim
        
        return pose_matrix

    def keypoint_matcher(self, pose_matrix=None, keypoint_threshold=0.005, match_threshold=0.20, image_height=480, image_width=640):

        # Check Pose Matrix Shape
        if pose_matrix.shape != (1, 4, 4):
            print("ERROR: Invalid Pose Matrix -- Incorrect Shape")

        # Render NeRF image bassed on pose_matrix and save in localizer images folder
        NeRF_Renderer().nerf_image(pose_matrix, image_height, image_width)

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


        keypoint_data = np.load(os.path.join(output_dir, "nerf_render_robot_image_matches.npz"))
        # print('----------------------------', np.sum(keypoint_data['keypoints0']>-1), '----------------------------')
        
        # Extract keypoints from the first and second images
        keypoints0 = keypoint_data["keypoints0"]  # Shape: [N0, 2] (x, y) keypoints in image 0
        keypoints1 = keypoint_data["keypoints1"]  # Shape: [N1, 2] (x, y) keypoints in image 1

        # Extract match indices (-1 means no match)
        matches = keypoint_data["matches"]  # Shape: [N0] - each index points to keypoints1 or -1 if no match

        # Filter out invalid matches (where matches[i] == -1)
        valid_matches = matches > -1

        if np.sum(valid_matches) == 0:
            print("No Keypoints Matched with Enough Confidance for Localization")
            return

        matched_kpts0 = keypoints0[valid_matches]  # Get matched keypoints in image 0
        matched_kpts1 = keypoints1[matches[valid_matches]]  # Get corresponding keypoints in image 1

        matched_kpts0 = torch.tensor(matched_kpts0).to(self.device)
        matched_kpts1 = torch.tensor(matched_kpts1).to(self.device)

        # Print some results
        print(f"Valid Matches: {len(matched_kpts0)}")
        # print("Image 0 Keypoints:", matched_kpts0)
        # print("Image 1 Keypoints:", matched_kpts1)

        return matched_kpts0, matched_kpts1

    def localize_pose(self, x=0.1, y=0.1, z=0.1, roll=0.05, pitch=0.05, yaw=0.05):

         # Create optimizable pose parameters to query NeRF -- Start at small non-zero values to avoid local minima
        pose_parameters = torch.tensor([x, y, z, pitch, roll, yaw], device='cuda', requires_grad=True)

        # Initialize optimizer
        optimizer = torch.optim.SGD([pose_parameters], lr=self.learning_rate, weight_decay=1e-3) # Use Classic gradient descent with L2 Regularization (weight decay)
        
        if self.render:
            # Create nerf_renders directory to store NeRF Renders
            optimizer_render_dir = os.path.join(os.getcwd(), "optimizer_renders")
            os.makedirs(optimizer_render_dir, exist_ok=True)

        # Optimization loop -- Keep optimizing until loss is below threshold or max iterations reached
        iter = 0
        loss_value = np.inf
        while iter < self.max_iters:

            if loss_value < self.loss_threshold:
                print("Loss Value Below Threshold: Pose Optimized")
                break

            pose_matrix = self.build_pose_matrix(pose_parameters)

            matched_keypoints0, matched_keypoints1 = self.keypoint_matcher(pose_matrix)

            break
            # keypoints_loss = torch.nn.functional.MSELoss() 
            
            iter += 1




        
        
# Test
Localizer().localize_pose(x=0.1, y=0.1, z=0.1, roll=0.05, pitch=0.05, yaw=0.05)


# Localizer().build_pose(image_path)



