import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time
import numpy as np

import datetime

# DO morphological ops
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("webcam_sub")
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, "/zed/zed_node/rgb_raw/image_raw_color", self.image_callback , qos_profile_sensor_data)
        self.counter = 0
    def image_callback(self, msg):
        try:
            start = time.time()
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')


            # gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            # hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # scaling raw hsv values to fit in opencv bounds
            # lower = np.array([231.43 / 2, 4.96 /100 * 255, 55.29 / 100 * 255]) - 20
            # upper = np.array([231.43 / 2, 4.96 / 100 * 255, 55.29 / 100 * 255]) + 20
            
            # lower = np.array([0, 0, 0])
            # upper = np.array([179, 255, 100]) 

            # lower = np.array([0, 4.96 /100 * 255, 55.29 / 100 * 255]) - 20
            # upper = np.array([255, 4.96 / 100 * 255, 55.29 / 100 * 255]) + 20
            
            
            # thresh = cv2.inRange(hsv, lower, upper, cv2.THRESH_BINARY_INV)


            # kernel_size = 5
            # blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            # low_t = 50
            # high_t = 150
            # edges = cv2.Canny(blur, low_t, high_t)

            # contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # print(contours)
            # for contour in contours:
            #     if len(contour) >= 1:
            #         print(contour)
            #         cv2.drawContours(cv_image, [contour], 0, (0,255,0), 5)
                    
            # if len(contours) >=1:
                # cv2.drawContours(cv_image, contours, -1, (0,255,0), 5)
            # cv2.imshow("hsv", hsv)
            # cv2.imshow("thresh", thresh)
            # cv2.imshow("Blurred", blur)
            # cv2.imshow('gray', gray)
            # cv2.imshow("Edges", edges)
            cv2.imshow('Webcam Stream', cv_image)

            print("image shown", time.time() - start)

            cv2.waitKey(1)
            current_time = datetime.datetime.now()

            # file_name =  "imgs/img_" + str(current_time.microsecond)
            file_name =  "imgs/img_" + str(self.counter) + ".jpg"
            self.counter += 1
            print(file_name + "\n")
            cv2.imwrite(file_name, cv_image)
            # counter += 1
        except Exception as e:
            self.get_logger().error('cv_bridge exception: %s' % e)


def main(args=None):
    rclpy.init(args=args)

    ip = ImageSubscriber()
    print("Subscribing...")
    rclpy.spin(ip)

    ip.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()