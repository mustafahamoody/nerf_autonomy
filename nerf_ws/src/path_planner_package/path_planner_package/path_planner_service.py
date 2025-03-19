import rclpy
from rclpy.node import Node
import numpy as np

from nav_msgs.msg import OccupancyGrid # For Subscription
from nav_msgs.srv import GetPlan # For Service and Path Publishing
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

# Path Planner Options: A* (A Star), RRT* (Rapidly-exploring Random Tree Star)
from path_planner_package.path_planners.path_planner_Astar_cm import a_star_cm
from path_planner_package.path_planners.path_planner_Astar import a_star
from path_planner_package.path_planners.path_planner_Astar_bad import a_star_bad
from path_planner_package.path_planners.path_planner_RRTstar import rrt_star


path_planner = a_star_cm # Choose Path Planner to use: a_star (A*) or rrt_star (RRT*)

class PathPlannerService(Node):
    def __init__(self):
        super().__init__('path_planner_service')
        
        # Subscriber for occupancy grid
        self.occupancy_grid_subscriber = self.create_subscription(OccupancyGrid, 'costmap', self.get_occupancy_grid, 10)
        
        # Service Server to publish path to controller (with start, goal, and headding parameters)
        self.service = self.create_service(GetPlan, 'get_path', self.get_path_callback)

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

    def get_occupancy_grid(self, msg):

        if self.occupancy_grid_received:
            return

        # Get grid from subscription
        width, height = msg.info.width, msg.info.height
        data = np.array(msg.data).reshape((height, width))
        self.grid = np.where(data == -1, 100, data)  # Replace unknown (-1) with high cost (100)
        self.get_logger().info('Occupancy grid received')

        # Occupancy grid details (from message)
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

        self.occupancy_grid_received = True

    def get_path_callback(self, request, response):
                     
        if self.grid is None:
            self.get_logger().warn("Waiting for occupancy grid...")
            return  # Exit until the grid is received
        
        # Get start and goal from service client
        start = (float(request.start.pose.position.x), float(request.start.pose.position.y))
        goal = (float(request.goal.pose.position.x), float(request.goal.pose.position.y))

        # Convert start and goal world coordinates to grid indices
        start_x = int((start[0] - self.origin_x) / self.resolution)
        start_y = int((start[1] - self.origin_y) / self.resolution)
        goal_x = int((goal[0] - self.origin_x) / self.resolution)
        goal_y = int((goal[1] - self.origin_y) / self.resolution)

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

        self.get_logger().info(f'-------------Recived Path request from current position: {start} to goal: {goal}-------------')
        
        # Get path: Run path planner on grid
        path = path_planner(self.grid, start, goal)
        
        if not path:
            self.get_logger().error("No path found!")
            return
        self.get_logger().info(f"Path found with {len(path)} steps.")

        # Convert grid indices back to world coordinates.
        # Assume the center of a cell (i, j) is:
        # wx = origin_x + (i + 0.5) * resolution, similarly for y.
        path_world = []
        for i, j in path:
            wx = self.origin_x + (i + 0.5) * self.resolution
            wy = self.origin_y + (j + 0.5) * self.resolution
            path_world.append((wx, wy))

        if path_world:
            # Create Response and Response plan (path) objects
            response = GetPlan.Response()
            response.plan = Path()

            # Response Header
            response.plan.header.stamp = self.get_clock().now().to_msg()
            response.plan.header.frame_id = 'map'

            for x, y in path_world:
                pose = PoseStamped()
                pose.header = response.plan.header

                pose.pose.position.x = float(x)
                pose.pose.position.y = float(y)
                pose.pose.position.z = 0.0

                response.plan.poses.append(pose)
         
        return response

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlannerService()
    rclpy.spin(planner)
    path_planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()