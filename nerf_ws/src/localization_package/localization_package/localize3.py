import torch
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import yaml
import os
import subprocess
from pathlib import Path
import cv2

# Class to render NeRF Image and return tensor 
from nerf_renderer import NeRFRenderer

# Class to run keypoint matching and detection
from keypoint_matcher import KeypointMatcher


# Utility to check autograd graph
# from torchviz import make_dot

########################### Load NERF Model Config. ###########################

class Localizer():
    def __init__(self, robot_image_path=None, learning_rate=0.01, max_iters=1000, keypoint_threshold=0.005, match_threshold=0.80, min_match=5, render=False, stream=False):
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
        self.image_size = [640, 480]

        # Path to robot image
        robot_image_path = os.path.join(os.getcwd(), "localizer_images", "robot_image.png") if robot_image_path == None else robot_image_path

        # Initalize NeRF Renderer class to query NeRF
        self.nerf_render = NeRFRenderer(save_image=render)

        # Initialize Keypoint Matcher class
        self.keypoint_threshold = keypoint_threshold
        self.match_threshold = match_threshold
        self.min_match = min_match
        self.keypoint_matcher = KeypointMatcher(robot_image_path, resize=self.image_size, viz=render, keypoint_threshold=self.keypoint_threshold, match_threshold=self.match_threshold)


    def build_pose_matrix(self, pose_parameters):
        x, y, z, roll, pitch, yaw = pose_parameters[0], pose_parameters[1], pose_parameters[2], pose_parameters[3], pose_parameters[4], pose_parameters[5]

        # Calculate trig values
        cos_roll, sin_roll = torch.cos(roll), torch.sin(roll)
        cos_pitch, sin_pitch = torch.cos(pitch), torch.sin(pitch)
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)

        # Create rotation matrices while preserving gradients
        Rz = torch.zeros((3, 3), device='cuda', dtype=torch.float32)
        Rz[0, 0] = 1.0
        Rz[1, 1] = cos_yaw
        Rz[1, 2] = -sin_yaw
        Rz[2, 1] = sin_yaw
        Rz[2, 2] = cos_yaw

        Ry = torch.zeros((3, 3), device='cuda', dtype=torch.float32)
        Ry[0, 0] = -cos_pitch
        Ry[0, 2] = sin_pitch
        Ry[1, 1] = -1.0
        Ry[2, 0] = -sin_pitch
        Ry[2, 2] = -cos_pitch

        Rx = torch.zeros((3, 3), device='cuda', dtype=torch.float32)
        Rx[0, 0] = cos_roll
        Rx[0, 1] = -sin_roll
        Rx[1, 0] = sin_roll
        Rx[1, 1] = cos_roll
        Rx[2, 2] = 1.0

        # Matrix multiplication while preserving gradients
        R = torch.matmul(torch.matmul(Rz, Ry), Rx)

        # Create translation part
        t = torch.zeros((3, 1), device='cuda', dtype=torch.float32)
        t[0, 0] = x
        t[1, 0] = y
        t[2, 0] = z

        # Create transformation matrix
        top_rows = torch.cat([R, t], dim=1)
        bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device='cuda')
        pose_matrix = torch.cat([top_rows, bottom_row], dim=0).unsqueeze(0)
        
        return pose_matrix


    def localize_pose(self, x=0.1, y=0.1, z=0.1, roll=0.05, pitch=0.05, yaw=0.05):

        # Create optimizable pose parameters to query NeRF -- Start at small non-zero values to avoid local minima, if no best-guess pose available
        pose_parameters = torch.tensor([x, y, z, pitch, roll, yaw], device='cuda', requires_grad=True)

        # Use Adam optimizer for better handling of different scales
        optimizer = torch.optim.Adam([pose_parameters], lr=0.01)

        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            verbose=True, 
            min_lr=1e-6
        )
        
        # For loss smoothing
        loss_history = []
        smoothing_window = 3
        best_loss = float('inf')
        best_params = pose_parameters.clone().detach()

        if self.render:
            # Create nerf_renders directory to store NeRF Renders
            optimizer_render_dir = os.path.join(os.getcwd(), "optimizer_renders")
            os.makedirs(optimizer_render_dir, exist_ok=True)


        # Optimization loop -- Keep optimizing until loss is below threshold or max iterations reached
        iter = 0
        while iter < self.max_iters:
            optimizer.zero_grad()

            # Build pose matrix from pose_parameters
            pose_matrix = self.build_pose_matrix(pose_parameters)
            # print('Pose Matrix Requires Grad', pose_matrix.requires_grad)


            # Render NeRF image bassed on pose_matrix and return as tensor
            nerf_render_tensor = self.nerf_render.nerf_image(pose_matrix, 
                                                            image_width=self.image_size[0], 
                                                            image_height=self.image_size[1], 
                                                            iter=iter)
            # print('NeRF Render Requires Grad', nerf_render_tensor.requires_grad)
            exit()

            # Run keypoint detection with SuperPoint and keypoint matching with SuperGlue and return matched keypoint coordinates (robot, nerf image respectivly)
            matched_keypoints0, matched_keypoints1, robot_image_tensor = self.keypoint_matcher.detect_and_match_keypoints(nerf_render_tensor, iter)


            # Modify pose and skip itteration if no keypoints matched
            if None in [matched_keypoints0, matched_keypoints1] or len(matched_keypoints0) == 0:
                        print("No keypoint matches found - skipping iteration")
                        # Small random perturbation to parameters to escape local minimum
                        with torch.no_grad():
                            pose_parameters.data += torch.randn_like(pose_parameters) * 0.05
                        iter += 1
                        continue

            # Calculate normalized loss: Euclidian Distance between matched keypoints / number of keypoint matches matches (for consistancy)
            # If not divided by # keypoints: More keypoints (desired for matched pose) = heigher loss
            num_matches = matched_keypoints0.shape[0]
            loss_keypoints = torch.sum(torch.norm(matched_keypoints0 - matched_keypoints1, dim=1) ** 2) / max(1, num_matches)

            # # Add Loss Peniltly for Blury NeRF Image ------ Need to implement with torch
            # nerf_render_path = os.path.join(Path.cwd(), "localizer_images", "nerf_render.png")
            # nerf_render = cv2.imread(nerf_render_path, cv2.IMREAD_GRAYSCALE)
            # laplacian_var = cv2.Laplacian(nerf_render, cv2.CV_64F).var() # Calculate image blur using Laplacian variance
            # print("NeRF Render Laplacian Variance:", laplacian_var)
            # print("Render is", "sharp" if laplacian_var >= 200 else "blury") 

            # # Inverse Normalization of variance
            # blur_penalty = lambda_blur * (1 / (1 + laplacian_var))
            # print(blur_penalty)
            # print("Loss Penelty : ", loss_keypoints)

            # # Calculate total loss
            # loss = loss_keypoints + laplacian_var
            # print('Pre Optimization Loss Tensor:', loss)

            # Update loss history for smoothing
            current_loss = loss_keypoints.item()
            loss_history.append(current_loss)
            if len(loss_history) > smoothing_window:
                loss_history.pop(0)

            smoothed_loss = sum(loss_history) / len(loss_history)

            # Track best parameters
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = pose_parameters.clone().detach()

            # Check for convergence
            if smoothed_loss < self.loss_threshold:
                print(f"Loss below threshold ({smoothed_loss} < {self.loss_threshold}): Pose optimized")
                pose_parameters.data.copy_(best_params)
                break


            # Backward pass
            loss_keypoints.backward()
            optimizer.step()

            # Gradient diagnostics
            if pose_parameters.grad is not None:
                grad_mag = pose_parameters.grad.abs().mean().item()
                print(f"Gradient magnitude: {grad_mag:.10f}")
            
                 # Scale up tiny gradients if needed, to ensure meaningful updates
                if grad_mag < 1e-6:
                    scale_factor = 1e-6 / max(grad_mag, 1e-10)
                    pose_parameters.grad.mul_(scale_factor)
                    print(f"Scaling gradients by {scale_factor:.6f}")

                # Clip large gradients to ensure smooth optimization process
                torch.nn.utils.clip_grad_norm_([pose_parameters], max_norm=1.0)

            # Print progress
            print(f"--------------------------Iteration {iter}, Loss:{current_loss:.4f} (Smoothed: {smoothed_loss:.4f})--------------------------")
            print(f"Current Pose Parameters: {pose_parameters.data}")
            print(f"Gradients: {pose_parameters.grad if pose_parameters.grad is not None else 'None'}")

            # Update parameters
            optimizer.step()
        
            # Update learning rate based on smoothed loss
            if len(loss_history) == smoothing_window:
                scheduler.step(smoothed_loss)
        
            iter += 1
    
        # Return best found parameters
        pose_parameters.data.copy_(best_params)
        return pose_parameters.detach().cpu().numpy()



# Test -- ROTATIONS BROKEN. Yaw controlls roll. Pitch dosen't work. NO YAW
Localizer(render=True, learning_rate=100, max_iters=100, keypoint_threshold=0.005, match_threshold=0.80, min_match=1
          ).localize_pose(x=0.1, y=0.1, z=0.1, roll=0.00, pitch=0.00, yaw=0.00)


# Localizer().build_pose(image_path)



