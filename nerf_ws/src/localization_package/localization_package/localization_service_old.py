import rclpy
from rclpy.node import Node
from rclpy.service import Service

import os
import cv2
from cv_bridge import CvBridge
import tf_transformations as tf

from sensor_msgs.msg import Image
from nav_msgs.msg import Pose

# NeRF-bassed Postion Localizer
from localization_package.localization_package.localize import Localize


class LocalizationService(Node):
    def __init__(self):
        super().__init__('localization_service')

        # Service Server to get send pose (x, y, z) and yaw from localizer
        self.service = self.create_service(Pose, 'localize_pose', self.localize_callback)



    def localize_callback(self, request, response):
        # Approximate pose values from controller
        approx_x = request.position.x
        approx_y = request.position.y
        approx_z = request.position.z
        approx_yaw = request.position.yaw

        self.get_logger().warn(f'Recived Position Query from approximate Pose ({approx_x}, {approx_y}, {approx_z}) and orientation {approx_yaw}')
        
        # Run Localizer
        translation, yaw = Localize().run(image_path, approx_x, approx_y, approx_z, approx_yaw)

        self.get_logger().warn(f"Localized Pose: ({translation[0]}, {translation[1]}, {translation[2]}, yaw: {yaw})")
        
        # Extract Pose values from Localizer
        response.position.x = translation[0]
        response.position.y = translation[1]
        response.position.z = translation[2]
        response.orientation.z = yaw

        return response

def main(args=None):
    rclpy.init(args=args)
    localizer = LocalizationService()
    rclpy.spin(localizer)
    localizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()