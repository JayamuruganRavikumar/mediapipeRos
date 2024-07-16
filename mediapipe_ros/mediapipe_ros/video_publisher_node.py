import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class VideoPublisher(Node):

    def __init__(self):
        super().__init__('video_publisher')
        self.rgb_publisher_ = self.create_publisher(Image, 'rgb/image_raw', 10)
        self.depth_publisher_ = self.create_publisher(Image, 'depth/image_raw', 10)
        self.timer_ = self.create_timer(0.1, self.timer_callback)
        self.cap = cv2.VideoCapture('video/test_video.mp4')
        self.bridge = CvBridge()

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        self.rgb_publisher_.publish(msg)
        self.depth_publisher_.publish(msg)
        self.get_logger().info('Publishing video frame')

def main(args=None):
    rclpy.init(args=args)
    video_publisher = VideoPublisher()
    rclpy.spin(video_publisher)
    video_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

