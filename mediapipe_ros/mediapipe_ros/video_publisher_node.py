import rclpy
import os
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
from ament_index_python.packages import get_package_share_directory

class VideoPublisher(Node):

    def __init__(self):
        super().__init__('video_publisher')
        self.rgb_publisher_ = self.create_publisher(Image, 'rgb/image_raw', 10)
        self.depth_publisher_ = self.create_publisher(Image, 'depth/image_raw', 10)
        package_dir = get_package_share_directory('mediapipe_ros')
        video_path = os.path.join(package_dir, 'video', 'test_video.mp4')
        self.cap = cv2.VideoCapture(video_path)
        self.bridge = CvBridge()

        if not self.cap.isOpened():
            self.get_logger().info('vide no')
        else:
            self.get_logger().info('vide open')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps==0:
            fps=30

        self.timer_ = self.create_timer(1/fps, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            if isinstance(frame, np.ndarray):

                msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
                self.rgb_publisher_.publish(msg)
                self.depth_publisher_.publish(msg)
                self.get_logger().info('Publishing video frame')
            else:
                self.get_logger().info('not numpu')
        else:
            self.get_logger().warning('end')
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def main(args=None):
    rclpy.init(args=args)
    video_publisher = VideoPublisher()
    rclpy.spin(video_publisher)
    video_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

