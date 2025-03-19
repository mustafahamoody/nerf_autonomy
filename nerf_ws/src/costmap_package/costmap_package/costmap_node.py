#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
from scipy.ndimage import distance_transform_edt

class CostmapNode(Node):
    def __init__(self):
        super().__init__('costmap_node')
        
        # Declare runtime parameters
        self.declare_parameter('decay_rate', 1.0)         # Exponential decay constant - how much cost decreases with distance
        self.declare_parameter('inflation_radius', 4)       # Inflation radius in cells - how far the costmap extends around obstacles
        self.declare_parameter('cost_threshold', 1.0)       # Minimum cost threshold - below this value, cost is set to 0
        
        # Subscriber: receive the raw occupancy grid
        self.occ_sub = self.create_subscription(
            OccupancyGrid,
            'occupancy_grid_2d',
            self.occ_callback,
            10
        )
        
        # Publisher: output the inflated costmap as an OccupancyGrid
        self.costmap_pub = self.create_publisher(
            OccupancyGrid,
            'costmap',
            10
        )
        
        self.get_logger().info("Costmap node initialized.")

    def occ_callback(self, occ_msg: OccupancyGrid):
        # Retrieve current parameters (allows for runtime changes)
        decay_rate = self.get_parameter('decay_rate').value
        inflation_radius = self.get_parameter('inflation_radius').value
        cost_threshold = self.get_parameter('cost_threshold').value
        
        width = occ_msg.info.width
        height = occ_msg.info.height
        resolution = occ_msg.info.resolution

        # Convert the 1D occupancy grid data into a 2D NumPy array
        grid = np.array(occ_msg.data).reshape((height, width))
        
        # Create a binary grid for the distance transform:
        # free cells (value 0) become 1, obstacles (value 100) become 0.
        binary_grid = np.where(grid == 100, 0, 1)
        
        # Compute the Euclidean distance from each free cell to the nearest obstacle.
        # The distances are in cell units.
        distances = distance_transform_edt(binary_grid)
        
        # Initialize a cost grid (floating point) with the same shape as the occupancy grid.
        cost_grid = np.zeros_like(grid, dtype=np.float32)
        
        # Obstacles remain at a cost of 100.
        obstacles_mask = (grid == 100)
        cost_grid[obstacles_mask] = 100
        
        # For free cells, compute the cost using an exponential decay profile.
        # cost = 100 * exp(-decay_rate * distance)
        free_mask = ~obstacles_mask
        computed_costs = 100 * np.exp(-decay_rate * distances)
        
        # For cells beyond the inflation radius, set the cost to 0.
        computed_costs[distances > inflation_radius] = 0
        
        # Apply the threshold: if the computed cost is below the threshold, set it to 0.
        computed_costs[computed_costs < cost_threshold] = 0
        
        # Update the cost grid for free cells.
        cost_grid[free_mask] = computed_costs[free_mask]
        
        # Convert the cost grid to int8 since OccupancyGrid data is an array of int8.
        cost_grid_int = cost_grid.astype(np.int8)
        
        # Flatten the 2D grid into a 1D list as required by the OccupancyGrid message.
        cost_data = cost_grid_int.flatten().tolist()
        
        # Create and populate the costmap message.
        costmap_msg = OccupancyGrid()
        costmap_msg.header = occ_msg.header     # Use the same frame and timestamp as the incoming grid
        costmap_msg.info = occ_msg.info         # Reuse the map metadata (resolution, origin, etc.)
        costmap_msg.data = cost_data
        
        # Publish the costmap.
        self.costmap_pub.publish(costmap_msg)
        self.get_logger().info("Published updated costmap.")

def main(args=None):
    rclpy.init(args=args)
    node = CostmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
