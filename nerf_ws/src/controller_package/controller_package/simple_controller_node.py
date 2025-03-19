import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped
import time
import numpy as np

class PathExecutor(Node):
    def __init__(self):
        super().__init__('simple_controller_node')

        # Subscribe to the planned path
        self.subsciber_ = self.create_subscription(Path, 'path', self.path_callback, 10)

        # Publisher for robot velocity commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Timer to periodically execute movement commands
        self.timer = self.create_timer(0.5, self.move_robot)

        # store path and execution state
        self.path = []
        self.current_waypoint_index = 0
        self.robot_x, self.robot_y = 0.0, 0.0
        self.robot_yaw = 0.0


    def path_callback(self, msg):
        # Recive path from path planner
        self.get_logger().info(f'Recived New Path with {len(msg.poses)} Waypoints')

        self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.current_waypoint_index = 1  # Start target at second waypoint

        # Start at first waypoint
        if self.path:
            self.robot_x, self.robot_y = self.path[0]


    def move_robot(self):
        # Move robot along planned path

        if not self.path or self.current_waypoint_index >= len(self.path):
            self.get_logger().info('Path completed or no path available')
            return  # No path available or path completed
        
        # Set robot target to next waypoint
        target_x, target_y = self.path[self.current_waypoint_index]
        
        # Compute x, y and norm distance to target
        dx = target_x - self.robot_x
        dy = target_y - self.robot_y
        distance = np.linalg.norm([dx, dy])

        # Compute angle to target
        target_yaw = np.arctan2(dy, dx)
        angle_diff = target_yaw - self.robot_yaw

        # Normalise angle to [-pi, pi]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        # Create velocity command
        twist = Twist()

        # Rotate towards the waypoint first
        if abs(angle_diff) > 0.05: 
            twist.angular.z = 1.0 * angle_diff # Increase angular speed for faster turns
            twist.linear.x = 0.0  # Don't move forward while turning
            self.get_logger().info(f'Rotating: {twist.angular.z:.2f}')
            self.cmd_pub.publish(twist)

            rotate_time = abs(angle_diff / twist.angular.z) 
            time.sleep(rotate_time) # sleep for t = d/v seconds -- Wait for rotation to complete
            
            twist.angular.z = 0.0 # Stop rotating
            self.cmd_pub.publish(twist)

        # Move forward
        twist.linear.x = min(0.5, distance) 
        self.get_logger().info(f'Moving: {twist.linear.x:.2f}')
        self.cmd_pub.publish(twist)

        move_time = distance / twist.linear.x
        time.sleep(move_time) # Wait for the robot to reach the waypoint

        twist.linear.x = 0.0  # Stop moving
        self.cmd_pub.publish(twist)

        self.get_logger().warn('Waypoint Reached')


        # Update Robot Internal States -- Assuming perfect mouvment execution (issue in practice, but fine for initial test)
        self.robot_x, self.robot_y = target_x, target_y
        self.robot_yaw = target_yaw

        # Target Next Waypoint
        self.current_waypoint_index += 1 # Only increment index after fully reaching the waypoint


def main(args=None):
    rclpy.init(args=args)
    executor = PathExecutor()
    rclpy.spin(executor)
    executor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()