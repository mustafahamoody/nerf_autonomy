#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
import numpy as np
import math
import yaml
import os

# ROS message imports
from visualization_msgs.msg import Marker
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header

# Load configurations from YAML file
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

# Get config paths from environment variables with error checking
occupancy_config_path = os.environ.get('OCCUPANCY_CONFIG_PATH')

if not occupancy_config_path:
    raise EnvironmentError("OCCUPANCY_CONFIG_PATH environment variable must be set")
    
# Load configurations
try:
    config_occupancy = load_config(occupancy_config_path)

except Exception as e:
    raise e

class GroundSegmentationNode(Node):
    def __init__(self):
        super().__init__('ground_segmentation_node')
        
        # Get parameter values
        self.ground_lower = config_occupancy['segmentation']['ground_lower']
        self.ground_upper = config_occupancy['segmentation']['ground_upper']
        self.z_center = config_occupancy['segmentation']['z_center']
        self.z_band_thickness = config_occupancy['segmentation']['z_band_thickness']
        self.grid_res = int(config_occupancy['segmentation']['grid_resolution'])
        self.grid_min = config_occupancy['segmentation']['grid_min']
        self.grid_max = config_occupancy['segmentation']['grid_max']
        self.voxel_size = config_occupancy['segmentation']['voxel_size']


        # Publisher for the 2D occupancy grid
        self.occ_pub = self.create_publisher(OccupancyGrid, 'occupancy_grid_2d', 10)

        # Subscribe to the voxel_marker topic (3D occupancy grid as a Marker message)
        self.sub = self.create_subscription(
            Marker,
            'voxel_marker',
            self.voxel_marker_callback,
            10)

        self.get_logger().info("GroundSegmentationNode initialized.")

    def _filter_isolated_cells(self, occ_2d):
        """
        Filter out isolated occupied cells in the 2D occupancy grid.
        For each cell with value 100, count the 8 neighbors (cardinal and diagonal).
        If the count is 0, set that cell to 0.
        Returns a new filtered 2D grid.
        """
        filtered = occ_2d.copy()
        rows, cols = occ_2d.shape
        for i in range(rows):
            for j in range(cols):
                if occ_2d[i, j] == 100:
                    count = 0
                    # Iterate over the 8 neighboring cells
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue  # Skip the cell itself
                            ni = i + di
                            nj = j + dj
                            if 0 <= ni < rows and 0 <= nj < cols:
                                if occ_2d[ni, nj] == 100:
                                    count += 1
                    # If there are no occupied neighbors, mark the cell as free.
                    if count == 0:
                        filtered[i, j] = 0
        return filtered

    def voxel_marker_callback(self, marker_msg: Marker):
        """
        Callback that processes the incoming 3D occupancy grid (Marker message),
        segments out the ground, projects a horizontal band to 2D, and publishes an OccupancyGrid.
        """
        # Create a full 3D occupancy array (binary) with shape (grid_res, grid_res, grid_res)
        grid_shape = (self.grid_res, self.grid_res, self.grid_res)
        occ_3d = np.zeros(grid_shape, dtype=np.uint8)  # 0: free, 1: occupied

        # We assume the Marker message contains a list of geometry_msgs/Point in marker_msg.points,
        # each representing the center of an occupied voxel.
        # The grid covers [grid_min, grid_max] along each axis and is uniformly spaced.
        for pt in marker_msg.points:
            # Extract x, y, z coordinates from the point (assumed to be float already)
            x = pt.x
            y = pt.y
            z = pt.z
            # Compute grid indices using the known domain and voxel size.
            # The centers of voxels are at: grid_min + (i + 0.5) * voxel_size for i=0...grid_res-1.
            i = int(round(((x - self.grid_min) / self.voxel_size) - 0.5))
            j = int(round(((y - self.grid_min) / self.voxel_size) - 0.5))
            k = int(round(((z - self.grid_min) / self.voxel_size) - 0.5))
            # Ensure indices are within bounds
            if 0 <= i < self.grid_res and 0 <= j < self.grid_res and 0 <= k < self.grid_res:
                occ_3d[i, j, k] = 1

        # Ground segmentation: set voxels whose z-coordinate is within [ground_lower, ground_upper] to free.
        # For each z index, compute the center z coordinate.
        for k in range(self.grid_res):
            center_z = self.grid_min + (k + 0.5) * self.voxel_size
            if self.ground_lower <= center_z <= self.ground_upper:
                occ_3d[:, :, k] = 0  # clear entire layer

        # 2D Projection:
        # Define the band: we use a horizontal band around z_center.
        z_band_min = self.z_center - self.z_band_thickness / 2.0
        z_band_max = self.z_center + self.z_band_thickness / 2.0

        # Create an empty 2D occupancy grid (binary) with dimensions (grid_res, grid_res)
        occ_2d = np.zeros((self.grid_res, self.grid_res), dtype=np.uint8)
        # For each (i,j) cell, check all k values (z) where the voxel center is within the band.
        for i in range(self.grid_res):
            for j in range(self.grid_res):
                occupied_in_band = False
                for k in range(self.grid_res):
                    center_z = self.grid_min + (k + 0.5) * self.voxel_size
                    if z_band_min <= center_z <= z_band_max:
                        if occ_3d[i, j, k] == 1:
                            occupied_in_band = True
                            break
                if occupied_in_band:
                    occ_2d[i, j] = 100  # Occupied (using standard ROS occupancy values: 100 for occupied)
                else:
                    occ_2d[i, j] = 0  # Free

        occ_2d = np.flipud(occ_2d)

        # Now rotate 90 degrees counter-clockwise.
        occ_2d = np.rot90(occ_2d, k=-1)

        occ_2d = self._filter_isolated_cells(occ_2d) #Filtering isolated cells

        # Create and publish the OccupancyGrid message
        occ_msg = OccupancyGrid()
        occ_msg.header = marker_msg.header  # use the same frame and time as the incoming marker
        # Set the map metadata
        meta = MapMetaData()
        meta.map_load_time = self.get_clock().now().to_msg()
        meta.resolution = self.voxel_size  # each cell is voxel_size meters
        meta.width = self.grid_res
        meta.height = self.grid_res
        # The origin of the occupancy grid: bottom-left corner.
        # In our grid, the x and y domain is from grid_min to grid_max.
        # We set z=0 for the 2D map.
        meta.origin.position.x = self.grid_min
        meta.origin.position.y = self.grid_min
        meta.origin.position.z = 0.0
        meta.origin.orientation.w = 1.0  # no rotation
        occ_msg.info = meta

        # The data for OccupancyGrid is a 1D list (row-major order).
        # We flatten the 2D occupancy grid (convert numpy array to a list of ints).
        occ_msg.data = occ_2d.flatten().tolist()

        self.occ_pub.publish(occ_msg)
        self.get_logger().info("Published 2D occupancy grid.")

def main(args=None):
    rclpy.init(args=args)
    node = GroundSegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
