import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time
import numpy as np

# DO morphological ops
counter = 0
hMin = 0
sMin = 0
vMin = 100
hMax = 255
sMax = 255
vMax = 255

height = .7

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("webcam_sub")
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, "/zed/zed_node/rgb_raw/image_raw_color", self.image_callback , qos_profile_sensor_data)

    def image_callback(self, msg):
        try:
            start = time.time()
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')


            # Set minimum and max HSV values to display
            lower = np.array([hMin, sMin, vMin])
            upper = np.array([hMax, sMax, vMax])

            # Create HSV Image and threshold into a range.
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            output = cv2.bitwise_and(img,img, mask= mask)

            # Print if there is a change in HSV value
            # if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            #     print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            #     phMin = hMin
            #     psMin = sMin
            #     pvMin = vMin
            #     phMax = hMax
            #     psMax = sMax
            #     pvMax = vMax

            # Morphological operations
            
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
            open_and_close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)


            # Guassian Blur and Canny Edges
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            kernel_size = 5
            blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            low_t = 50
            high_t = 150
            edges = cv2.Canny(blur, low_t, high_t)

            cv_image = output.copy()

            # Create a mask with only bottom region
            blank_mask = np.zeros_like(cv_image)
            ignore_mask_color = (255,255,255)
            rows, cols = cv_image.shape[:2]
            bottom_left  = [cols * 0, rows * 1]
            top_left     = [cols * 0, rows * height]
            bottom_right = [cols * 1, rows * 1 ]
            top_right    = [cols * 1, rows * height]

            # smaller mask
            # top_left     = [cols * 0, rows * 0.7]
            # top_right    = [cols * 1, rows * 0.7]


            vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
            # filling the polygon with white color and generating the final mask
            cv2.fillPoly(blank_mask, vertices, ignore_mask_color)
            cv2.imshow("mask", blank_mask)


                
            # performing Bitwise AND on the input image and mask to get only the edges on the road
            masked_image = cv2.bitwise_and(output, blank_mask).astype(np.uint8)
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
            
            # eroded = cv2.morphologyEx(masked_image, cv2.MORPH_OPEN, kernel)

            cv2.imshow("masked_img", masked_image)
            print(masked_image.dtype)
            # thresh_image = thresh_image.astype(np.uint8)

            contours_canny,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            mask_contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # not using morphological operations

            # contours_open,_ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # contours_close,_ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # contours_open_and_close,_ = cv2.findContours(open_and_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # draw contours on base guassian image
            if len(contours_canny) >=1:
                cv2.drawContours(cv_image, contours_canny, -1, (0,255,0), 5)
                for contour in contours_canny:
                    if cv2.contourArea(contour) > 200:
                        print(cv2.contourArea(contour))
                    
                        x,y,w,h = cv2.boundingRect(contour)
                        print("X: " , x , "\tY: ", y, "\tW: " , w ,"\tH: ", h)
                        cv2.rectangle(cv_image, (x,y), (x+w,y+h), (0,0,255), 1)
            
            cpy_img = output.copy()
            cpy_img_rt = output.copy()
            cpy_img_circle = output.copy()

            # draw contours on masked image
            if len(mask_contours) >=1:
                    cv2.drawContours(cpy_img, mask_contours, -1, (255,255,0), 5)
                    left_contours = []
                    right_contours = []
                    # loopp through mask_contours and append to left and right countours arrays
                    for contour in mask_contours:
                        if cv2.contourArea(contour) > 150:
                            centroid, dimensions, angle = cv2.minAreaRect(contour)
                            if (centroid[0] < rows / 2):
                                left_contours.append(contour)
                            else:
                                right_contours.append(contour)
                            
                            # draw rotated rect
                            rect = cv2.minAreaRect(contour)
                            box = cv2.boxPoints(rect)
                            box = np.int0(box)
                            cv2.drawContours(cpy_img_rt,[box],0,(0,0,255),2)

                            # draw circle
                            (x,y),radius = cv2.minEnclosingCircle(contour)
                            center = (int(x),int(y))
                            radius = int(radius)
                            cv2.circle(cpy_img_circle,center,radius,(255,0,0),2)

                            print(cv2.contourArea(contour))

                            # draw rectangle
                            x,y,w,h = cv2.boundingRect(contour)
                            print("X: " , x , "\tY: ", y, "\tW: " , w ,"\tH: ", h)
                            cv2.rectangle(cpy_img, (x,y), (x+w,y+h), (0,0,255), 1)
                    # draw midpoint between biggest left and right contour
                    if left_contours and right_contours:
                        max_left_contour = left_contours[0]
                        max_right_contour = right_contours[0]
                        for right_contour in right_contours:
                            if(cv2.contourArea(right_contour) > cv2.contourArea(max_right_contour)):
                                max_right_contour = right_contour
                        for left_contour in left_contours:
                            if(cv2.contourArea(left_contour) > cv2.contourArea(max_left_contour)):
                                max_left_contour = left_contour
                        centroid_left, dimensions_left, angle_left = cv2.minAreaRect(max_left_contour)
                        centroid_right, dimensions_right, angle_right = cv2.minAreaRect(max_right_contour)
                        cv2.circle(cpy_img_rt, (int(centroid_right[0]), int(centroid_right[1])), 5, (36,255,12), -1)
                        cv2.circle(cpy_img_rt, (int(centroid_left[0]), int(centroid_left[1])), 5, (36,255,12), -1)
                        midpoint_x = (centroid_right[0] + centroid_left[0]) / 2
                        midpoint_y = (centroid_right[1] + centroid_left[1]) / 2
                        cv2.circle(cpy_img_rt, ((int(midpoint_x)), int(midpoint_y)), 5, (255,0,0), -1)
                        print("midpoint: " , midpoint_x,",", midpoint_y)

            # show images
            cv2.imshow('image',output)
            # cv2.imshow('canny', cv_image)
            # cv2.imshow('canny mask', cpy_img)
            cv2.imshow('rotate rect mask', cpy_img_rt)
            # cv2.imshow('rotate circle mask', cpy_img_circle)

                    

            # print("image shown", time.time() - start)

            cv2.waitKey(1)
            # file_name =  "imgs/" + counter
            # print(file_name + "\n")
            # cv2.imwrite(file_name, cv_image)
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