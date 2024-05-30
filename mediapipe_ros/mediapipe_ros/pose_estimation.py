import rclpy
import cv2
import mediapipe as mp
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from mediapipe_msg.msg import PoseList
from mediapipe.python.solutions.pose import PoseLandmark
from sensor_msgs.msg import Image
# Use message_filters to synchronize the depth and rgb camera
from message_filters import ApproximateTimeSynchronizer, Subscriber


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
NAME_POSE = [
    (PoseLandmark.NOSE), (PoseLandmark.LEFT_EYE_INNER),
    (PoseLandmark.LEFT_EYE), (PoseLandmark.LEFT_EYE_OUTER),
    (PoseLandmark.RIGHT_EYE_INNER), ( PoseLandmark.RIGHT_EYE),
    (PoseLandmark.RIGHT_EYE_OUTER), ( PoseLandmark.LEFT_EAR),
    (PoseLandmark.RIGHT_EAR), ( PoseLandmark.MOUTH_LEFT),
    (PoseLandmark.MOUTH_RIGHT), ( PoseLandmark.LEFT_SHOULDER),
    (PoseLandmark.RIGHT_SHOULDER), ( PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW), ( PoseLandmark.LEFT_WRIST),
    (PoseLandmark.RIGHT_WRIST), ( PoseLandmark.LEFT_PINKY),
    (PoseLandmark.RIGHT_PINKY), ( PoseLandmark.LEFT_INDEX),
    (PoseLandmark.RIGHT_INDEX), ( PoseLandmark.LEFT_THUMB),
    (PoseLandmark.RIGHT_THUMB), ( PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_HIP), ( PoseLandmark.LEFT_KNEE),
    (PoseLandmark.RIGHT_KNEE), ( PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.RIGHT_ANKLE), ( PoseLandmark.LEFT_HEEL),
    (PoseLandmark.RIGHT_HEEL), ( PoseLandmark.LEFT_FOOT_INDEX),
    (PoseLandmark.RIGHT_FOOT_INDEX)
]

class PosePublisher(Node):

    def __init__(self):
        super().__init__('mediapipe_pose_publisher')
        self.publisher_ = self.create_publisher(PoseList, '/mediapipe/pose_list', 10)
        self.image_publisher=self.create_publisher(Image, '/processed/image', 10)
        self.sub_rgb = Subscriber(Image, '/rgb/image_raw')
        self.sub_depth = Subscriber(Image, '/depth_to_rgb/image_raw')
        self.ts = ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.getdepth_callback, self.getrgb_callback)
        self.bridge = CvBridge()

    #callback function for depth camera    
    def getcamera_callback(self, msg):
        try:
            image_msg = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.rgb = cv2.cvtColor(cv2.flip(image_msg, 0), cv2.COLOR_BGR2RGB)                
            #conver form 16UC1 to np array
            #depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height,msg.width)                 
            depth = self.bridge.imgmsg_to_cv2(msg, "16UC1") 
            #flip the depth image
            self.depth = depth[::-1,:]
            self.compare_depth(self.rgb,self.depth)

        except CvBridgeError as e:
            self.get_logger().error(f"Error converting from depth camera: {str{e}}")
        except Exception as e:
            self.get_logger().error(f"Camera : {e}")

        self.get_logger().error(f"Synced depth camera")

    #compare depth and rgb image
    def compare_depth(self, image, depth):

        poselist = PoseList() 
        
        with mp_pose.Pose(
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5) as pose:
            image.flags.writeable = False
            results = pose.process(image)
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks( image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            h, w, c = image.shape

            if results.pose_landmarks != None:
                index = 0
                for ids, pose_landmarks in enumerate(results.pose_landmarks.landmark):
                    #check for wrist
                    if 15 <= ids < 23:
                        cx,cy = pose_landmarks.x*w, pose_landmarks.y*h
                        poselist.human_pose[index].name = str(NAME_POSE[ids])
                        poselist.human_pose[index].x = cx
                        poselist.human_pose[index].y = cy
                        poselist.human_pose[index].z = float(depth[int(cy),int(cx)])
                        index+=1
                self.publisher_.publish(poselist)

            else: 
                index = 0
                for ids, pose_landmarks in enumerate(results.pose_landmarks.landmark):
                    if 15 <= ids < 23:
                        poselist.human_pose[index].name = str(NAME_POSE[ids])
                        poselist.human_pose[index].x = 0.0
                        poselist.human_pose[index].y = 0.0
                        poselist.human_pose[index].z = 0.0
                        index+=1
                self.publisher_.publish(poselist)


            img_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.image_publisher.publish(img_msg)

def main(args=None):

    rclpy.init(args=args)
    pose_publisher=PosePublisher()
    rclpy.spin(pose_publisher)
    pose_publisher.destroy_node()


if __name__ == '__main__':
    main()

