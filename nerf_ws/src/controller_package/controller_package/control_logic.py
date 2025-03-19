from geometry_msgs.msg import Twist
import numpy as np

class RobotController():
    def __init__(self, node):
        # Use publisher, logger and timer from parent Node
        self.node = node
        self.publish = node.cmd_pub.publish  
        
        # Robot Start Position 
        self.robot_x, self.robot_y = self.node.robot_x, self.node.robot_y
        self.robot_yaw = self.node.robot_yaw

        # State Variables
        self.state = 'IDLE'  # Possible states: IDLE, ROTATING, MOVING, REACHED WAYPOINT
        self.target_x = None
        self.target_y = None
        self.target_yaw = None

        # Distance to target
        self.distance = None
        self.angle_diff = None

        # Motion Parameters
        self.angular_speed = 1.0  # Angular speed
        self.move_speed = 2.0  # Linear speed
        self.update_rate = 0.1 # Timer update rate in seconds

        # Check if robot started moving before updating position
        self.current_angular_speed = None
        self.current_linear_speed = None

        # Timer to check current position (non-blocking control execution)
        self.timer = self.node.create_timer(self.update_rate, self.control_loop)


    def target(self, target_x, target_y): 
        """ Update Target Waypoints and begin motion """
        
        self.robot_x, self.robot_y = self.node.robot_x, self.node.robot_y
        self.robot_yaw = self.node.robot_yaw

        self.target_x = target_x
        self.target_y = target_y

        self.get_angle_diff()


        if abs(self.angle_diff) > 0.05:
            self.state = 'ROTATING'
            self.node.get_logger().warn(f'Robot: {self.state} (angle_diff={self.angle_diff:.2f})')
            self.rotate()

        else:
            self.state = 'MOVING'
            self.node.get_logger().warn(f'Robot: {self.state} (distance_diff={self.distance:.2f})')
            self.move()


    def control_loop(self):
        """ Main controller loop to regularly check and update robot state and position """

        if self.state == 'ROTATING':
            
            # Update Robot Yaw, if it already started rotating
            if self.current_angular_speed:
                yaw_change = self.current_angular_speed * self.update_rate
                self.robot_yaw += yaw_change
                self.robot_yaw = (self.robot_yaw + np.pi) % (2 * np.pi) - np.pi  # Normalize yaw
                self.get_angle_diff()

            if abs(self.angle_diff) < 0.05:
                self.stop_move()
                self.state = 'MOVING'
                self.node.get_logger().warn(f'Robot: {self.state}')
                self.current_angular_speed = None
                self.move()

        elif self.state == 'MOVING':
            
            # Update Robot Position, if already started moving
            if self.current_linear_speed:
                self.robot_x += self.current_linear_speed * np.cos(self.robot_yaw) * self.update_rate
                self.robot_y += self.current_linear_speed * np.sin(self.robot_yaw) * self.update_rate
                self.get_distance()

            if self.distance < 0.1:
                self.stop_move()
                self.state = 'REACHED WAYPOINT'
                self.node.get_logger().warn(f'Robot: {self.state}')
                self.current_linear_speed = None


        elif self.state == 'REACHED WAYPOINT':
            self.state = 'IDLE'
            self.node.robot_x, self.node.robot_y, self.node.robot_yaw = self.target_x, self.target_y, self.target_yaw
            self.node.reached_waypoint()

        else:
            self.node.get_logger().warn('Waiting for Path')


    def rotate(self):
        """ Rotate Robot towards target waypoint """
        twist = Twist()
        self.current_angular_speed = self.angular_speed * self.angle_diff  # Increase angular speed for bigger turns
        twist.angular.z = self.current_angular_speed
        self.publish(twist)

        
    def move(self):
        """ Move Robot towards target waypoint """
        twist = Twist()
        self.current_linear_speed = min(self.move_speed, self.distance)
        twist.linear.x = self.current_linear_speed
        self.publish(twist)
    

    def stop_move(self):

        twist = Twist() 
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publish(twist)


    def get_distance(self):
        "Compute x, y and norm distance to target waypoint"
        dx = self.target_x - self.robot_x
        dy = self.target_y - self.robot_y
        self.distance = np.linalg.norm([dx, dy])

        return dx, dy


    def get_angle_diff(self):
        
        # Need distance to waypoint to compute angle
        dx, dy = self.get_distance()

        "Compute angle to target waypoint"
        self.target_yaw = np.atan2(dy, dx)
        angle_diff = self.target_yaw - self.robot_yaw

        # Normalise angle (yaw) to [-pi, pi]
        self.angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

    def kill(self):
        quit()
