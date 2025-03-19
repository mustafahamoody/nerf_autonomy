import rclpy
from rclpy.node import Node

from nav_msgs.srv import GetPlan
from geometry_msgs.msg import PoseStamped, Twist

import numpy as np

from controller_package.control_logic import RobotController # Import controller logic


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')

        # Set start and goal positions
        self.start = (-0.5, -0.9)  # (y, x)
        self.goal = (0.9, 0.9)  # (y, x)

        # Service Client to request path from path planner
        self.client = self.create_client(GetPlan, 'get_path')

        # Publisher for robot velocity commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Robot (Initial) Start Position 
        self.robot_x, self.robot_y = self.start[0], self.start[1]
        self.robot_yaw = 0.0
        
        # Initialize Robot Controller
        self.controller = RobotController(self)
        
        #Set Start and Goal Positions
        self.start = PoseStamped()
        self.start.pose.position.x = self.robot_x
        self.start.pose.position.y = self.robot_y

        self.goal = PoseStamped()
        self.goal.pose.position.x = self.goal[0]
        self.goal.pose.position.y = self.goal[1]

        # Sets target waypoint and stores path before calling path planner again
        self.target_waypoint_index = 0
        self.old_path = []
        self.path = []

        # Request path from planner
        self.request_plan()


    def request_plan(self):
        """ Request path from planner service"""
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for Path Planner')
        
        request = GetPlan.Request()
        request.start = self.start
        request.goal = self.goal

        # Call Path Planner and wait until path is retuned
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        # Store Service Response and use to make path list
        plan = future.result()
        self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in plan.plan.poses]
        
        if not self.path:
            self.get_logger().error('Failed to Retrive Path from Planner')
            return
        
        if len(self.old_path) < len(self.path) and self.target_waypoint_index != 0: #Older path is more optimal and not first run
            self.path = self.old_path
            self.get_logger().warn(f'Using same path: Optimal')
            self.target_waypoint_index += 1  # Set target to next waypoint
            self.move_to_waypoint()
        else:
            self.get_logger().warn(f'Recived Path with {len(self.path)} Waypoints')
            # Start Moving to Target Waypoint
            self.target_waypoint_index = 1  # Set target to second waypoint
            self.move_to_waypoint()
    

    def move_to_waypoint(self):
        """Move to Target Waypoint"""

        # Stop if path is complete
        if self.target_waypoint_index < len(self.path):

            # Get target positon
            target_x, target_y = self.path[self.target_waypoint_index]

            # Call controller
            self.controller.target(target_x, target_y)

            # if self.reached_goal():
            #     self.get_logger().error('Goal Reached')
            #     return
            # else:
            #      self.old_path = self.path
            #      self.request_plan()
        
        else:
            self.get_logger().error('Path completed')
            self.controller.kill()
            return
    
    def reached_waypoint(self):
        """ Callback from controller when robot reaches waypoint"""
        self.target_waypoint_index += 1
        self.move_to_waypoint()

    # def reached_goal(self):  
    #     goal_x = self.goal.pose.position.x
    #     goal_y = self.goal.pose.position.y

    #     dx = self.robot_x - goal_x
    #     dy = self.robot_y - goal_y

    #     goal_distance = np.linalg.norm([dx, dy])
    #     return goal_distance < 0.1
    

def main(args=None):
    rclpy.init(args=args)
    controller = ControllerNode()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main
