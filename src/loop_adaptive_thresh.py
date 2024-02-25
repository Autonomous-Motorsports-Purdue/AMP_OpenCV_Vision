import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
def nothing(x):
    pass

# Create a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','image',0,255,nothing)
cv2.createTrackbar('VMin','image',0,255,nothing)
cv2.createTrackbar('HMax','image',0,179,nothing)
cv2.createTrackbar('SMax','image',0,255,nothing)
cv2.createTrackbar('VMax','image',0,255,nothing)
cv2.createTrackbar('blockSizeGaus','image',3,400,lambda x: x if x % 2 == 1 else x + 1)
cv2.createTrackbar('blockSizeMean','image',3,400,lambda x: x if x % 2 == 1 else x + 1)
cv2.createTrackbar('constantGaus','image',-255,255,nothing)
cv2.createTrackbar('constantMean','image',-255,255,nothing)




# fix trackbar, currently does nothing
cv2.createTrackbar('Height','image',0,100,nothing)


# Set default value for MAX HSV trackbars.

# Left side line
# cv2.setTrackbarPos('HMax', 'image', 179)
# cv2.setTrackbarPos('SMax', 'image', 26)
# cv2.setTrackbarPos('VMax', 'image', 255)

# cv2.setTrackbarPos('VMin', 'image', 158)

cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 26)
cv2.setTrackbarPos('VMax', 'image', 255)

cv2.setTrackbarPos('VMin', 'image', 158)
cv2.setTrackbarPos('Height', 'image', 70)
cv2.setTrackbarPos('Height', 'image', 70)



# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

# img = cv2.imread('img.jpg')
img = cv2.imread('imgs/img_7.jpg')

blockSizeGaus = 117
blockSizeMean = 117
constantGaus = -25
constantMean = -25
closing_iterations = 1
kernel_size = 5
height = .7
# 55 -22
# 31 -25
output = img

waitTime = 0
height = 0
img_count = 0
while(1):
    if img_count > 116:
        img_count = 0
    if img_count < 0:
        img_count = 116
    
    file_name = "imgs/img_" + str(img_count) + ".jpg"
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)


    blank_mask = np.zeros_like(img)
    ignore_mask_color = (255,255,255)
    rows, cols = img.shape[:2]
    if height < 0:
        height = 0
    if height > 1:
        height = 1
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
    masked_image = cv2.bitwise_and(img, blank_mask).astype(np.uint8)
    cv2.imshow("masked_img", masked_image)    




    # eroded = cv2.morphologyEx(masked_image, cv2.MORPH_OPEN, kernel)

    # cv2.imshow("masked_img", masked_image)
    


    img_normal = cv2.imread(file_name)

    hMin = cv2.getTrackbarPos('HMin','image')
    
    # blockSizeGaus = cv2.getTrackbarPos('blockSizeGaus','image')
    # blockSizeMean = cv2.getTrackbarPos('blockSizeMean','image')
    # constantGaus = cv2.getTrackbarPos('constantGaus','image')
    # constantMean = cv2.getTrackbarPos('constantMean','image')
    
    # if blockSizeGaus % 2 == 1:
    #     print(blockSizeGaus)
    # elif blockSizeMean % 2 == 1:
    #     print(blockSizeMean)
    # elif blockSizeGaus % 2 == 0:
    #     blockSizeGaus = (blockSizeGaus // 2)  + 1
    # elif blockSizeMean % 2 == 0:
    #     blockSizeMean = (blockSizeMean // 2)  + 1
    
    img = cv2.medianBlur(img,5)
    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    print(blockSizeMean)
    print(blockSizeGaus)

    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,blockSizeGaus,constantGaus)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,blockSizeMean,constantMean)
    

    th_masked = cv2.adaptiveThreshold(masked_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,blockSizeMean,constantMean)
    
    
    if kernel_size < 0:
        kernel_size = 1
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    opening = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)
    erosion = cv2.erode(th3,kernel,iterations = 1)
    closing = cv2.morphologyEx(th3,cv2.MORPH_CLOSE,kernel)

    closing_many = cv2.morphologyEx(th3,cv2.MORPH_CLOSE,kernel, iterations = closing_iterations)
    closing_many_masked = cv2.morphologyEx(th_masked,cv2.MORPH_CLOSE,kernel, iterations = closing_iterations)


    # invert image
    invert = cv2.bitwise_not(th3)

    closing_invert = cv2.morphologyEx(invert, cv2.MORPH_CLOSE, kernel)
    erosion_invert = cv2.erode(invert,kernel,iterations = 1)
    
    contours_invert, _ = cv2.findContours(invert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cpy_contours_invert = img_normal.copy()

    cpy_img = img_normal.copy()
    cpy_img_all = img_normal.copy()

    if len(contours_invert) >=1:
        cv2.drawContours(cpy_contours_invert, contours_invert, -1, (0,0,255), 5)
        for contour in contours_invert:
            if cv2.contourArea(contour) > 100:
                centroid, dimensions, angle = cv2.minAreaRect(contour)
                # if (centroid[0] < rows / 2):
                #     left_contours.append(contour)
                # else:
                #     right_contours.append(contour)
                
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(cpy_img,[box],0,(255,0,0),2)

    contours_gauss,_ = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours_gauss) >=1:
        cv2.drawContours(cpy_img_all, contours_gauss, -1, (0,0,255), 5)
        left_contours = []
        right_contours = []
        for contour in contours_gauss:
            if cv2.contourArea(contour) > 1:
                centroid, dimensions, angle = cv2.minAreaRect(contour)
                # if (centroid[0] < rows / 2):
                #     left_contours.append(contour)
                # else:
                #     right_contours.append(contour)
                
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(cpy_img,[box],0,(0,0,255),2)
    
    cv2.imshow('Original Image', img)
    # cv2.imshow('Global Thresholding (v = 127)', th1)
    cv2.imshow('Adaptive Mean Thresholding', th2)
    cv2.imshow('Adaptive Gaussian Thresholding', th3)
    # cv2.imshow('Opening', opening)
    # cv2.imshow('Erosion', erosion)
    # cv2.imshow('Closing', closing)
    # cv2.imshow('Closing', closing)
    # cv2.imshow('Closing Many', closing_many)


    cv2.imshow('masked', th_masked)
    cv2.imshow('closing many masked', closing_many_masked)
    

    # cv2.imshow('guassian contours', cpy_img)
    # cv2.imshow('guassian contours_all', cpy_img_all)
    # cv2.imshow('invert', invert)
    # cv2.imshow('closing invert', closing_invert)
    # cv2.imshow('erosion invert', erosion_invert)


    # cv2.imshow('contours invert', cpy_contours_invert)






    """
    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')

    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

    height = int(cv2.getTrackbarPos('Height','image')) / 100
    # height = 0.7

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img,img, mask= mask)

    # Print if there is a change in HSV value
    if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

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

    # get only bottom region
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
    
    opening = cv2.morphologyEx(masked_image, cv2.MORPH_OPEN, kernel)

    cv2.imshow("masked_img", masked_image)
    
    print(masked_image.dtype)
    # thresh_image = thresh_image.astype(np.uint8)

    contours_canny,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask_contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # contours_open,_ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contours_close,_ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contours_open_and_close,_ = cv2.findContours(open_and_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # print(contours)
    # for contour in contours:
    #     if len(contour) >= 1:
    #         print(contour)
    #         cv2.drawContours(cv_image, [contour], 0, (0,255,0), 5)
            
    if len(contours_canny) >=1:
        cv2.drawContours(cv_image, contours_canny, -1, (0,255,0), 5)
        for contour in contours_canny:
            if cv2.contourArea(contour) > 0:
                print(cv2.contourArea(contour))
                centroid, dimensions, angle = cv2.minAreaRect(contour)
                # if (centroid[0] < rows / 2):
                #     left_contours.append(contour)
                # else:
                #     right_contours.append(contour)
                
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(cv_image,[box],0,(0,0,255),2)
              
                # x,y,w,h = cv2.boundingRect(contour)
                # print("X: " , x , "\tY: ", y, "\tW: " , w ,"\tH: ", h)
                # cv2.rectangle(cv_image, (x,y), (x+w,y+h), (0,0,255), 1)
    
    cpy_img = output.copy()
    cpy_img_rt = output.copy()
    cpy_img_circle = output.copy()

    # draw contours on image, find midpoint between biggest left and right contours
    if len(mask_contours) >=1:
            cv2.drawContours(cpy_img, mask_contours, -1, (255,255,0), 5)
            left_contours = []
            right_contours = []
            for contour in mask_contours:
                if cv2.contourArea(contour) > 10:
                    centroid, dimensions, angle = cv2.minAreaRect(contour)
                    if (centroid[0] < rows / 2):
                        left_contours.append(contour)
                    else:
                        right_contours.append(contour)
                    
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(cpy_img_rt,[box],0,(0,0,255),2)

                    # centroid, dimensions, angle = cv2.minAreaRect(contour)
                    # cv2.circle(cpy_img_rt, (int(centroid[0]), int(centroid[1])), 5, (36,255,12), -1)

                    
                    (x,y),radius = cv2.minEnclosingCircle(contour)
                    center = (int(x),int(y))
                    radius = int(radius)
                    cv2.circle(cpy_img_circle,center,radius,(255,0,0),2)

                    print(cv2.contourArea(contour))
                
                    x,y,w,h = cv2.boundingRect(contour)
                    print("X: " , x , "\tY: ", y, "\tW: " , w ,"\tH: ", h)
                    cv2.rectangle(cpy_img, (x,y), (x+w,y+h), (0,0,255), 1)
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

    # Display input image
    print(file_name)
    cv2.imshow('input', img)

    # Display output image
    cv2.imshow('image',output)

    cv2.imshow('opening', opening)

    # cv2.imshow('closing', closing)
    # cv2.imshow('open_and_close', open_and_close)

    cv2.imshow('canny', cv_image)
    # cv2.imshow('mask_canny', cpy_img)

    # cv2.imshow('eroded', eroded)
    # cv2.imshow('')

    cv2.imshow('canny mask', cpy_img)
    cv2.imshow('rotate rect mask', cpy_img_rt)
    # cv2.imshow('rotate circle mask', cpy_img_circle)

    """

    # step through images
    if cv2.waitKey(waitTime) & 0xFF == ord('s'):
        img_count += 1
    if cv2.waitKey(waitTime) & 0xFF == ord('a'):
        img_count -= 1    
    # end program
    if cv2.waitKey(waitTime) & 0xFF == ord('q'):
        break
    if cv2.waitKey(waitTime) & 0xFF == ord('1'):
        blockSizeGaus += 2
        blockSizeMean += 2
        # print("*" * 24)
        # print("blockSizeGaus: ", blockSizeGaus)
        # print("blockSizeMean: ", blockSizeMean)

    elif cv2.waitKey(waitTime) & 0xFF == ord('2'):
        blockSizeGaus -= 2
        blockSizeMean -= 2
        # print("*" * 24)
        # print("blockSizeGaus: ", blockSizeGaus)
        # print("blockSizeMean: ", blockSizeMean)
    elif cv2.waitKey(waitTime) & 0xFF == ord('3'):
        constantGaus += 1
        constantMean += 1
    elif cv2.waitKey(waitTime) & 0xFF == ord('4'):
        constantGaus -= 1
        constantMean -= 1
    elif cv2.waitKey(waitTime) & 0xFF == ord('5'):
        closing_iterations += 1
    elif cv2.waitKey(waitTime) & 0xFF == ord('6'):
        closing_iterations -= 1
    elif cv2.waitKey(waitTime) & 0xFF == ord('7'):
        kernel_size += 1
    elif cv2.waitKey(waitTime) & 0xFF == ord('8'):
        kernel_size -= 1
    elif cv2.waitKey(waitTime) & 0xFF == ord('9'):
        height += 0.05
    elif cv2.waitKey(waitTime) & 0xFF == ord('0'):
        height -= 0.05
    print("blocksize: ", blockSizeGaus, "\tconstant: ", constantGaus, "\tclosing iterations: ", closing_iterations, "\tkernel size: ", kernel_size, "\theight: ", height)
    
    print(file_name)

cv2.destroyAllWindows()