import rclpy
from rclpy.node import Node
import numpy as np

from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

# Path Planner Options: A* (A Star), RRT* (Rapidly-exploring Random Tree Star)
from path_planner_package.path_planners.path_planner_Astar import a_star
from path_planner_package.path_planners.path_planner_Astar_bad import a_star_bad
from path_planner_package.path_planners.path_planner_RRTstar import rrt_star
from path_planner_package.path_planners.path_planner_Astar_cm import a_star_cm


# Choose Path Planner to use: a_star_cm (A* with cost map), a_star (A*), or rrt_star (RRT*)
path_planner = a_star_cm


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')
        
        # Set start and goal positions
        self.start = (-0.5, -0.9)  # (y, x)
        self.goal = (0.9, 0.9)  # (y, x)

        # Only publish path once
        self.publish_once = False


        # Subscriber for occupancy grid
        self.occupancy_grid_subscriber = self.create_subscription(OccupancyGrid, 'costmap', self.get_occupancy_grid, 10)
        
        # Publisher for path
        self.path_publisher_ = self.create_publisher(Path, 'path', 10)
        self.path_timer = self.create_timer(1.0, self.path_publisher_callback)
        
        # Object to store occupancy grid
        self.grid = None

        # Occupany grid details (from message) -- To setup start and goal
        self.width = None
        self.height = None
        self.resolution = None
        self.origin_x = None
        self.origin_y = None

        # Only get occupancy grid once (static map)
        self.occupancy_grid_received = False

        # Determine if path has already been published if publish_once is True
        self.path_published = False

    def get_occupancy_grid(self, msg):

        if self.occupancy_grid_received:
            return
        
        # Get grid from subscription
        width, height = msg.info.width, msg.info.height

        grid_data = np.array(msg.data).reshape((height, width))
        self.grid = np.where(grid_data == -1, 100, grid_data)  # Replace unknown (-1) with high cost (100)
        
        self.get_logger().info('Occupancy grid received')

        # Occupancy grid details (from message)
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

        self.occupancy_grid_received = True


    def path_publisher_callback(self): #msg):
        
        # Only publish path once
        if self.path_published == True:
            return
        
        if self.grid is None:
            self.get_logger().warn("Waiting for occupancy grid...")
            return  # Exit until the grid is received

        # Convert start and goal world coordinates to grid indices
        start_x = int((self.start[0] - self.origin_x) / self.resolution)
        start_y = int((self.start[1] - self.origin_y) / self.resolution)
        goal_x = int((self.goal[0] - self.origin_x) / self.resolution)
        goal_y = int((self.goal[1] - self.origin_y) / self.resolution)

        # Ensure indices are within bounds
        start_x = max(0, min(start_x, self.width - 1))
        start_y = max(0, min(start_y, self.height - 1))
        goal_x = max(0, min(goal_x, self.width - 1))
        goal_y = max(0, min(goal_y, self.height - 1))
        
        # Check if start or goal is in obstacle
        if self.grid[start_y, start_x] == 100:  # Note the order: [y,x]
            self.get_logger().error("Start position is in an obstacle!")
            return
        if self.grid[goal_y, goal_x] == 100:  # Note the order: [y,x]
            self.get_logger().error("Goal position is in an obstacle!")
            return

        start = (start_x, start_y)
        goal = (goal_x, goal_y)

        # Call path planner to create path
        path = path_planner(self.grid, start, goal) # Call path planner (selected above) to create path
        
        if not path:
            self.get_logger().error("No path found!")
            return
        self.get_logger().info(f"Path found with {len(path)} steps.")

        self.get_logger().info("Path grid indices: " + str(path))
        
        # Convert grid indices back to world coordinates.
        # Assume the center of a cell (i, j) is:
        # wx = origin_x + (i + 0.5) * resolution, similarly for y.
        path_world = []
        for i, j in path:
            wx = self.origin_x + (i + 0.5) * self.resolution
            wy = self.origin_y + (j + 0.5) * self.resolution
            path_world.append((wx, wy))

        if path_world:
            path_msg = Path()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = 'map'

            for x, y in path_world:
                pose = PoseStamped()
                pose.header = path_msg.header

                pose.pose.position.x = float(x)
                pose.pose.position.y = float(y)
                pose.pose.position.z = 0.0

                path_msg.poses.append(pose)
            self.path_publisher_.publish(path_msg)

            if self.publish_once:
                self.path_published = True


def main(args=None):
    rclpy.init(args=args)
    path_planner_node = PathPlannerNode()
    rclpy.spin(path_planner_node)
    path_planner_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()