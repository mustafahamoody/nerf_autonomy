#!/usr/bin/env python3

import sys
import argparse
from rclpy.utilities import remove_ros_args
import rclpy
from rclpy.node import Node
import numpy as np
import torch
import yaml
import os
import struct

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point  # Needed for Marker.CUBE_LIST

# Import your NeRFNetwork and Trainer
from nerf_config.libs.nerf.network import NeRFNetwork
from nerf_config.libs.nerf.utils import Trainer

#Import trainer options (opt)
from nerf_config.config.model_options import ModelOptions

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


class OccupancyGridNode(Node):
    def __init__(self, opt):
        super().__init__('occupancy_grid_node')
        
        # Get config paths from environment variables with error checking
        model_config_path = os.environ.get('MODEL_CONFIG_PATH')
        trainer_config_path = os.environ.get('TRAINER_CONFIG_PATH')
        
        if not model_config_path:
            raise EnvironmentError("MODEL_CONFIG_PATH environment variable must be set")
        if not trainer_config_path:
            raise EnvironmentError("TRAINER_CONFIG_PATH environment variable must be set")
            
        # Load configurations
        try:
            self.config_model = load_config(model_config_path)
            self.config_trainer = load_config(trainer_config_path)
            self.get_logger().info(f"Loaded model config from: {model_config_path}")
            self.get_logger().info(f"Loaded trainer config from: {trainer_config_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load configurations: {e}")
            raise


        # Publisher for the occupancy grid (PointCloud2) -- for future use.
        self.pc_pub = self.create_publisher(PointCloud2, 'occupancy_grid', 10)

        # Publisher for the 3D Marker (voxel grid visualization)
        self.marker_pub = self.create_publisher(Marker, 'voxel_marker', 10)

        # Two timers: one for testing a single point and one for the full occupancy grid.
        self.full_grid_timer = self.create_timer(5.0, self.publish_occupancy_grid)  # every 5 seconds

        # # Load configuration files via ROS parameters
        # self.declare_parameter('config_file_model',
        #                        '/home/kishorey/nerf_ws/src/occupancy_package/occupancy_package/config/model_config.yaml')
        # config_file_model = self.get_parameter('config_file_model').get_parameter_value().string_value
        # self.config_model = load_config(config_file_model)

        # self.declare_parameter('config_file_trainer',
        #                        '/home/kishorey/nerf_ws/src/occupancy_package/occupancy_package/config/trainer_config.yaml')
        # config_file_trainer = self.get_parameter('config_file_trainer').get_parameter_value().string_value
        # self.config_trainer = load_config(config_file_trainer)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.MSELoss(reduction='none')

        # Rotation matrix to align coordinate frames:
        # (Training: +z, Blender: -z)
        rot = torch.tensor([[0., 0., 1.],
                            [1., 0., 0.],
                            [0., 1., 0.]], device=self.device, dtype=torch.float32)
        self.rot = rot

        # Initialize the NeRFNetwork model
        self.model = NeRFNetwork(
            encoding=self.config_model['model']['encoding'],
            bound=self.config_model['model']['bound'],
            cuda_ray=self.config_model['model']['cuda_ray'],
            density_scale=self.config_model['model']['density_scale'],
            min_near=self.config_model['model']['min_near'],
            density_thresh=self.config_model['model']['density_thresh'],
            bg_radius=self.config_model['model']['bg_radius'],
        )
        self.model.eval()  # Set the model to evaluation mode

        # Initialize the Trainer (this loads weights from a checkpoint)
        self.trainer = Trainer(
            'ngp',
            opt=ModelOptions.opt(),
            model=self.model,
            device=self.device,
            workspace=self.config_trainer['trainer']['workspace'],
            criterion=self.criterion,
            fp16=self.config_model['model']['fp16'],
            use_checkpoint=self.config_trainer['trainer']['use_checkpoint'],
        )

        # Define a density function that applies the rotation and queries the model.
        # It returns the full dictionary output from self.model.density.
        self.nerf = lambda x: self.model.density(x.reshape((-1, 3)) @ self.rot)

    def create_cube_marker(self, position, scale=0.1, color=(0.0, 1.0, 0.0, 1.0), frame_id="map", marker_id=0):
        """Helper to create a cube marker at a given position (for single-point testing)."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "occupied_voxel"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.lifetime.sec = 0
        return marker

    def publish_occupancy_grid(self):
        """
        Generate a full 100x100x100 grid, compute the density (sigma) for each point,
        perform maxpooling to reduce the grid to 20x20x20, and publish a Marker.CUBE_LIST
        representing the occupied voxels.
        """
        side = 200
        # Generate grid coordinates in the domain [-1,1]
        x_linspace = torch.linspace(-1, 1, side, device=self.device)
        y_linspace = torch.linspace(-1, 1, side, device=self.device)
        z_linspace = torch.linspace(-1, 1, side, device=self.device)
        grid_coords = torch.stack(torch.meshgrid(x_linspace, y_linspace, z_linspace, indexing='ij'), dim=-1)
        grid_coords_flat = grid_coords.reshape(-1, 3)

        with torch.no_grad():
            result = self.nerf(grid_coords_flat)
        sigma = result['sigma']
        # Reshape sigma to the original 3D grid
        sigma_grid = sigma.reshape(side, side, side)
        # Add batch and channel dimensions for pooling: [1,1,100,100,100]
        sigma_grid_unsq = sigma_grid.unsqueeze(0).unsqueeze(0)
        kernel_size = 2  # This will reduce the grid to 20x20x20
        maxpool = torch.nn.MaxPool3d(kernel_size=kernel_size)
        sigma_pooled = maxpool(sigma_grid_unsq)[0, 0]  # Shape: [20, 20, 20]
        threshold = 15  # Threshold for occupied voxels
        occupied = sigma_pooled > threshold  # Boolean tensor

        # Compute voxel size: original domain length (2) divided by 20 = 0.1 per voxel.
        voxel_size = 2 / (side / kernel_size)  # 2/20 = 0.1

        # Compute center coordinates for each voxel in the reduced grid.
        # The centers along one axis: from -1 + voxel_size/2 to 1 - voxel_size/2.
        centers = torch.linspace(-1 + voxel_size / 2, 1 - voxel_size / 2, int(side / kernel_size), device=self.device)
        X, Y, Z = torch.meshgrid(centers, centers, centers, indexing='ij')
        center_grid = torch.stack((X, Y, Z), dim=-1)  # Shape: [20, 20, 20, 3]

        # Get the center coordinates of the occupied voxels.
        occupied_indices = occupied.nonzero(as_tuple=False)  # Shape: [N, 3]
        if occupied_indices.numel() == 0:
            self.get_logger().info("No occupied voxels found in the occupancy grid.")
            return
        occupied_centers = center_grid[occupied_indices[:, 0], occupied_indices[:, 1], occupied_indices[:, 2]]
        # occupied_centers: Tensor of shape [N, 3]

        # Create a Marker of type CUBE_LIST for efficiency.
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "occupancy_grid"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale.x = voxel_size
        marker.scale.y = voxel_size
        marker.scale.z = voxel_size
        # Set a uniform color (e.g., green)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime.sec = 0

        marker.points = []
        for center in occupied_centers.cpu().numpy():
            pt = Point()
            pt.x = float(center[0])
            pt.y = float(center[1])
            pt.z = float(center[2])
            marker.points.append(pt)

        self.marker_pub.publish(marker)
        self.get_logger().info(f"Published occupancy grid with {len(marker.points)} occupied voxels.")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space (linear, srgb)")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="use random pose sampling")
    args = remove_ros_args(sys.argv)
    opt = parser.parse_args(args[1:])  # Skip script name
    return opt


def main(args=None):
    rclpy.init(args=args)
    opt = parse_arguments()
    node = OccupancyGridNode(opt)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
