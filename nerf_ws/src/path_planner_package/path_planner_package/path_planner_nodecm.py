#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
import numpy as np
import heapq
from math import sqrt
from typing import List, Tuple, Dict

# ROS message imports
from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.msg import MapMetaData
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Header

##########################
# A* Helper Functions
##########################

def create_node(position: Tuple[int, int], g: float = float('inf'), 
                h: float = 0.0, parent: Dict = None) -> Dict:
    """
    Create a node for the A* algorithm.
    """
    return {
        'position': position,
        'g': g,
        'h': h,
        'f': g + h,
        'parent': parent
    }

def calculate_heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Calculate the Euclidean distance between two points.
    """
    x1, y1 = pos1
    x2, y2 = pos2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_valid_neighbors(grid: np.ndarray, position: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Get valid neighboring cells that are not obstacles.
    In this costmap, a cell with value 100 is considered an obstacle.
    Position is (x,y) but grid is accessed as [y,x].
    """
    x, y = position
    rows, cols = grid.shape
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= ny < rows and 0 <= nx < cols:
                # Skip if the neighbor cell is an obstacle.
                if grid[ny, nx] == 100:
                    continue
                # For diagonal moves, check that adjacent cardinal cells are not obstacles.
                if dx != 0 and dy != 0:
                    if grid[y + dy, x] == 100 or grid[y, x + dx] == 100:
                        continue
                neighbors.append((nx, ny))
    return neighbors

def reconstruct_path(goal_node: Dict) -> List[Tuple[int, int]]:
    """
    Reconstruct the path from goal to start by following parent pointers.
    """
    path = []
    current = goal_node
    while current is not None:
        path.append(current['position'])
        current = current['parent']
    return path[::-1]  # Reverse to get path from start to goal

def find_path(grid: np.ndarray, start: Tuple[int, int], 
              goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Find the optimal path using the A* algorithm.
    grid: 2D numpy array of cost values (0 = free, gradient values for near obstacles, 100 = obstacle)
    start and goal: grid indices (x, y)
    Returns a list of grid positions representing the path.
    """
    start_node = create_node(position=start, g=0, h=calculate_heuristic(start, goal))
    open_list = [(start_node['f'], start)]  # Priority queue of (f, position)
    open_dict = {start: start_node}
    closed_set = set()
    
    while open_list:
        _, current_pos = heapq.heappop(open_list)
        current_node = open_dict[current_pos]
        
        if current_pos == goal:
            return reconstruct_path(current_node)
            
        closed_set.add(current_pos)
        
        for neighbor_pos in get_valid_neighbors(grid, current_pos):
            if neighbor_pos in closed_set:
                continue
            # Incorporate both the movement cost (Euclidean distance) and the costmap penalty.
            move_cost = calculate_heuristic(current_pos, neighbor_pos)
            cell_penalty = float(grid[neighbor_pos[1], neighbor_pos[0]])
            tentative_g = current_node['g'] + move_cost + cell_penalty
            
            if neighbor_pos not in open_dict:
                neighbor = create_node(
                    position=neighbor_pos,
                    g=tentative_g,
                    h=calculate_heuristic(neighbor_pos, goal),
                    parent=current_node
                )
                heapq.heappush(open_list, (neighbor['f'], neighbor_pos))
                open_dict[neighbor_pos] = neighbor
            elif tentative_g < open_dict[neighbor_pos]['g']:
                neighbor = open_dict[neighbor_pos]
                neighbor['g'] = tentative_g
                neighbor['f'] = tentative_g + neighbor['h']
                neighbor['parent'] = current_node
                
    return []  # No path found

##########################
# ROS2 Path Planner Node
##########################

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')
        
        # Declare parameters for start and goal in world coordinates.
        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', -0.9)
        self.declare_parameter('goal_x', -0.9)
        self.declare_parameter('goal_y', 0.9)
        
        # Get the parameters.
        self.start_x = self.get_parameter('start_x').value
        self.start_y = self.get_parameter('start_y').value
        self.goal_x = self.get_parameter('goal_x').value
        self.goal_y = self.get_parameter('goal_y').value
        
        # Create a subscriber to the costmap.
        self.occ_sub = self.create_subscription(
            OccupancyGrid,
            'costmap',
            self.occ_grid_callback,
            10
        )
        
        # Publishers for the path and markers.
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        self.marker_pub = self.create_publisher(Marker, 'path_marker', 10)
        self.start_marker_pub = self.create_publisher(Marker, 'start_marker', 10)
        self.goal_marker_pub = self.create_publisher(Marker, 'goal_marker', 10)
        
        # To ensure we plan only once (if the costmap is static).
        self.path_planned = False
        
        self.get_logger().info("PathPlannerNode initialized.")

    def occ_grid_callback(self, occ_msg: OccupancyGrid):
        if self.path_planned:
            return
        
        self.get_logger().info("Received costmap message, planning path.")
        
        width = occ_msg.info.width
        height = occ_msg.info.height
        resolution = occ_msg.info.resolution
        origin_x = occ_msg.info.origin.position.x
        origin_y = occ_msg.info.origin.position.y
        
        # Reshape the 1D costmap data into a 2D NumPy array.
        grid_data = np.array(occ_msg.data).reshape((height, width))
        
        # Debug prints.
        self.get_logger().info(f"Grid shape: {grid_data.shape}")
        self.get_logger().info(f"Grid values range: min={np.min(grid_data)}, max={np.max(grid_data)}")
        self.get_logger().info(f"Number of obstacles: {np.sum(grid_data == 100)}")
        
        # Convert start and goal world coordinates to grid indices.
        start_x = int((self.start_x - origin_x) / resolution)
        start_y = int((self.start_y - origin_y) / resolution)
        goal_x = int((self.goal_x - origin_x) / resolution)
        goal_y = int((self.goal_y - origin_y) / resolution)
        
        # Ensure indices are within bounds.
        start_x = max(0, min(start_x, width - 1))
        start_y = max(0, min(start_y, height - 1))
        goal_x = max(0, min(goal_x, width - 1))
        goal_y = max(0, min(goal_y, height - 1))
        
        # Check if start or goal is in an obstacle.
        if grid_data[start_y, start_x] == 100:
            self.get_logger().error("Start position is in an obstacle!")
            return
        if grid_data[goal_y, goal_x] == 100:
            self.get_logger().error("Goal position is in an obstacle!")
            return
        
        start_idx = (start_x, start_y)
        goal_idx = (goal_x, goal_y)
        
        self.get_logger().info(f"Start grid index: {start_idx}, Goal grid index: {goal_idx}")
        
        # Log some obstacle positions for debugging.
        self.get_logger().info("Obstacle positions (first 5):")
        obstacle_positions = np.where(grid_data == 100)
        for y, x in zip(obstacle_positions[0][:5], obstacle_positions[1][:5]):
            self.get_logger().info(f"Obstacle at: ({x}, {y})")
        
        path_indices = find_path(grid_data, start_idx, goal_idx)
        
        if not path_indices:
            self.get_logger().error("No path found!")
            return
        self.get_logger().info(f"Path found with {len(path_indices)} steps.")
        self.get_logger().info("Path grid indices: " + str(path_indices))
        
        # Convert grid indices back to world coordinates.
        path_world = []
        for i, j in path_indices:
            wx = origin_x + (i + 0.5) * resolution
            wy = origin_y + (j + 0.5) * resolution
            path_world.append((wx, wy))
        
        # Publish the path as a nav_msgs/Path message.
        path_msg = Path()
        path_msg.header = occ_msg.header
        for (wx, wy) in path_world:
            pose = PoseStamped()
            pose.header = occ_msg.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # Identity orientation.
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)
        self.get_logger().info("Published nav_msgs/Path message.")
        
        # Publish a Marker message (LINE_STRIP) for visualizing the path.
        marker = Marker()
        marker.header = occ_msg.header
        marker.ns = "planned_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = resolution / 2.0  # Line width.
        marker.color.r = 90.0
        marker.color.g = 34.0
        marker.color.b = 139.0
        marker.color.a = 1.0
        for (wx, wy) in path_world:
            pt = Point()
            pt.x = wx
            pt.y = wy
            pt.z = 0.008
            marker.points.append(pt)
        self.marker_pub.publish(marker)
        self.get_logger().info("Published LINE_STRIP marker for path.")
        
        # Publish start and goal markers.
        self._publish_start_goal_markers(occ_msg.header, (self.start_x, self.start_y), (self.goal_x, self.goal_y), resolution)
        
        # Set flag so that we don't plan again.
        self.path_planned = True

    def _publish_start_goal_markers(self, header, start_world: Tuple[float, float], goal_world: Tuple[float, float], resolution: float):
        # Start marker (green sphere).
        start_marker = Marker()
        start_marker.header = header
        start_marker.ns = "start_goal"
        start_marker.id = 1
        start_marker.type = Marker.SPHERE
        start_marker.action = Marker.ADD
        start_marker.pose.position.x = start_world[0]
        start_marker.pose.position.y = start_world[1]
        start_marker.pose.position.z = 0.008
        start_marker.pose.orientation.w = 1.0
        start_marker.scale.x = resolution
        start_marker.scale.y = resolution
        start_marker.scale.z = resolution
        start_marker.color.r = 0.0
        start_marker.color.g = 1.0
        start_marker.color.b = 0.0
        start_marker.color.a = 1.0
        self.start_marker_pub.publish(start_marker)
        
        # Goal marker (red sphere).
        goal_marker = Marker()
        goal_marker.header = header
        goal_marker.ns = "start_goal"
        goal_marker.id = 2
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = goal_world[0]
        goal_marker.pose.position.y = goal_world[1]
        goal_marker.pose.position.z = 0.008
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = resolution
        goal_marker.scale.y = resolution
        goal_marker.scale.z = resolution
        goal_marker.color.r = 1.0
        goal_marker.color.g = 0.0
        goal_marker.color.b = 0.0
        goal_marker.color.a = 1.0
        self.goal_marker_pub.publish(goal_marker)
        
        self.get_logger().info("Published start and goal markers.")

def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()