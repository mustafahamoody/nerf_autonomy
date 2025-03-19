import rclpy
from rclpy.node import Node
from rclpy.task import future
import asyncio
import numpy as np

from geometry_msgs.msg import Pose
from sensor_msgs.srv import Image  # Assuming the image service returns a sensor_msgs/Image

from localization_package.localization_package.localize import Localize

class LocalizationService(Node):
    def __init__(self):
        super().__init__('localization_service')

        # Create localization service
        self.service = self.create_service(Pose, 'localized_pose', self.localize_callback)

        # Create client for camera image capture service
        self.image_client = self.create_client(Image, 'captured_image')

        self.get_logger().info('NeRF Localizer Ready')

    async def request_image(self):
        """
        Calls image capture service and returns the received image 
        """

        if not self.image_client.wait_for_service(timeout_secs=2.0):
            self.get_logger().error("Image Capture Service Not Available!")
            return None

        # Create empty request to tell imgage capture service to take and send picture
        request = Image.Request()

        # Asynchronously call the service
        future = self.image_client.call_async(request)
        await future 

        if future.result() is None:
            self.get_logger().error("Failed to Get Image From Capture Service!")
            return None

        return future.result().data # Assuming image data is stored in the 'data' field

    async def localize_callback(self, request, resposne_future: Future):
        """
        Processes localization requests from the controller and calls localizer
        """
        self.get_logger().info('Received Localization Request.')
        
        # Call image capture service asynchronously
        image_data = await self.request_image()

        if image_data is None:
            self.get_logger().error('Localization Failed Due to Missing Image.')
            resposne_future.set_result(Pose())  # Return empty pose
            return
        
        # Extract pose data if available
        x, y, z = request.position.x, request.position.y, request.position.z
        yaw = request.orientation.z


        # Call NeRF Localizer 
        final_translation, final_yaw = Localize().run(image_data, x, y, z, yaw)

        # Create response pose
        respone = Pose()
        respone.position.x = final_translation[0]
        respone.position.y = final_translation[1]
        respone.position.z = final_translation[2]
        respone.orientation.z = final_yaw

        # Return localized pose asynchronously
        resposne_future.set_result(respone)


def main(args=None):
    rclpy.init(args=args)
    localizer = LocalizationService()
    rclpy.spin(localizer)
    localizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
