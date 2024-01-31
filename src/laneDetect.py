import numpy as np
# import pandas as pd
import cv2

image = cv2.imread('img.jpg')

mask = np.zeros_like(image)   
# if you pass an image with more then one channel
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# applying gaussian Blur which removes noise from the image 
# and focuses on our region of interest
# size of gaussian kernel
kernel_size = 5
# Applying gaussian blur to remove noise from the frames
blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
# first threshold for the hysteresis procedure
low_t = 50
# second threshold for the hysteresis procedure 
high_t = 150
# applying canny edge detection and save edges in a variable
edges = cv2.Canny(blur, low_t, high_t)



mask = np.zeros_like(image)   
# if you pass an image with more then one channel
if len(image.shape) > 2:
    channel_count = image.shape[2]
    ignore_mask_color = (255,) * channel_count
# our image only has one channel so it will go under "else"
else:
        # color of the mask polygon (white)
    ignore_mask_color = 255
# creating a polygon to focus only on the road in the picture
# we have created this polygon in accordance to how the camera was placed
rows, cols = image.shape[:2]
bottom_left  = [cols * 0.1, rows * 0.95]
top_left     = [cols * 0.4, rows * 0.6]
bottom_right = [cols * 0.9, rows * 0.95]
top_right    = [cols * 0.6, rows * 0.6]
vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
# filling the polygon with white color and generating the final mask
cv2.fillPoly(mask, vertices, ignore_mask_color)
# performing Bitwise AND on the input image and mask to get only the edges on the road
masked_image = cv2.bitwise_and(image, mask)

# Distance resolution of the accumulator in pixels.
rho = 1             
# Angle resolution of the accumulator in radians.
theta = np.pi/180   
# Only lines that are greater than threshold will be returned.
threshold = 20      
# Line segments shorter than that are rejected.
minLineLength = 20  
# Maximum allowed gap between points on the same line to link them
maxLineGap = 500    
# function returns an array containing dimensions of straight lines 
# appearing in the input image
lines = cv2.HoughLinesP(edges, rho = rho, theta = theta, threshold = threshold,
                        minLineLength = minLineLength, maxLineGap = maxLineGap)


# region = region_selection(edges)
# Applying hough transform to get straight lines from our image 
# and find the lane lines
# Will explain Hough Transform in detail in further steps
# hough = hough_transform(region)
#lastly we draw the lines on our resulting frame and return it as output 
# result = draw_lane_lines(image, lane_lines(image, hough))


cv2.imshow('edges', edges)
cv2.imshow('region mask', masked_image)

    # Wait longer to prevent freeze for videos.
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
    
