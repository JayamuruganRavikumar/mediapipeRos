import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, TransformStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Header, ColorRGBA
from mediapipe_msg.msg import VisualPose, PoseList
from cv_bridge import CvBridge
import tf2_ros 

class Visualization(Node):
    def __init__(self):
        super().__init__('visualization')
        self.point_array_pub = self.create_publisher(VisualPose, '/mediapipe/points3D', 10)
        self.subscription=self.create_subscription(CameraInfo, '/mediapipe/rgb/camera_info', self.getcamerainfo_callback, 10)
        self.subscription=self.create_subscription(PoseList, 'mediapipe/pose_list', self.pixel_to_3d, 10)
        self.broadcaster = tf2_ros.TransformBroadcaster(self)
        self.bridge = CvBridge()
        self.caminfo = None


    def getcamerainfo_callback(self, msg):
        #self.get_logger().info("Received Cam info")
        try:
            self.caminfo = msg
        except Exception as e:
            self.getlogget().info(f"Camera Info error{e}")



    #Convert 
    def pixel_to_3d(self, msg):

        if self.caminfo is None:
            return

        points = VisualPose()
        source_frame = "camera_body"

        #Get camera intrinsic parameters
        # Camera matrix K = [fx 0 cx]
        #                   [0 fy cy]
        #                   [0 0  1 ]

        fx = self.caminfo.k[0]
        fy = self.caminfo.k[4]
        cx = self.caminfo.k[2]
        cy = self.caminfo.k[5]

        for i, val in enumerate(msg.human_pose):
            depth = val.z
            points.act_position[i].name =val.name
            points.act_position[i].x = float((val.x - cx) * depth / fx)
            points.act_position[i].y = float((val.y - cy) * depth / fy)
            points.act_position[i].z = float(depth)

        self.point_array_pub.publish(points)
        self.transform(points)

    def bounding_box(self, msg):
        #Get bounding box of the detected pose
        pass

    #View pose landmarks as tf
    def transform(self, points):
        source_frame = "camera_body"
        for i, point in enumerate(points.act_position):
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = source_frame
            transform.child_frame_id = point.name

            transform.transform.translation.x = point.x/1000
            transform.transform.translation.y = point.y/1000
            transform.transform.translation.z = point.z/1000
            transform.transform.rotation.w = 1.0
            self.broadcaster.sendTransform(transform)

def main(args=None):

    rclpy.init(args=args)
    visualization = Visualization()
    rclpy.spin(visualization)
    visualization.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
