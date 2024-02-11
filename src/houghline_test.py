import numpy as np
import cv2

def nothing(x):
    pass

cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','image',0,255,nothing)
cv2.createTrackbar('VMin','image',0,255,nothing)
cv2.createTrackbar('HMax','image',0,179,nothing)
cv2.createTrackbar('SMax','image',0,255,nothing)
cv2.createTrackbar('VMax','image',0,255,nothing)

# Set default value for MAX HSV trackbars.

# Left side line
# cv2.setTrackbarPos('HMax', 'image', 179)
# cv2.setTrackbarPos('SMax', 'image', 26)
# cv2.setTrackbarPos('VMax', 'image', 255)

# cv2.setTrackbarPos('VMin', 'image', 158)

cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 16)
cv2.setTrackbarPos('VMax', 'image', 204)

cv2.setTrackbarPos('VMin', 'image', 158)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

img = cv2.imread('imgs/img_0.jpg')
output = img
waitTime = 33

while(1):

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')

    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

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
    top_left     = [cols * 0, rows * 0.6735]
    bottom_right = [cols * 1, rows * 1 ]
    top_right    = [cols * 1, rows * 0.6735]
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
            if cv2.contourArea(contour) > 200:
                print(cv2.contourArea(contour))
              
                x,y,w,h = cv2.boundingRect(contour)
                print("X: " , x , "\tY: ", y, "\tW: " , w ,"\tH: ", h)
                cv2.rectangle(cv_image, (x,y), (x+w,y+h), (0,0,255), 1)

    cpy_img = output.copy()
    if len(mask_contours) >=1:
            cv2.drawContours(cpy_img, mask_contours, -1, (0,255,0), 5)
            for contour in contours_canny:
                if cv2.contourArea(contour) > 200:
                    print(cv2.contourArea(contour))
                
                    x,y,w,h = cv2.boundingRect(contour)
                    print("X: " , x , "\tY: ", y, "\tW: " , w ,"\tH: ", h)
                    cv2.rectangle(cpy_img, (x,y), (x+w,y+h), (0,0,255), 1)

    # if len(contours_open) >=1:
    #     cv2.drawContours(opening, contours_open, -1, (0,255,0), 5)
    # if len(contours_close) >=1:
    #     cv2.drawContours(closing, contours_close, -1, (0,255,0), 5)
    # if len(contours_open_and_close) >=1:
    #     cv2.drawContours(open_and_close, contours_open_and_close, -1, (0,255,0), 5)
    

    cny1 = cv2.Canny(output, 50, 200, None, 3)
    cdstP = cv2.cvtColor(cny1, cv2.COLOR_GRAY2BGR)
    
    linesP = cv2.HoughLinesP(cny1, 1, np.pi / 180, 50, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)


    # Display output image
    cv2.imshow('image',output)

    #cv2.imshow('opening', opening)
    #cv2.imshow('closing', closing)
    #cv2.imshow('open_and_close', open_and_close)

    #cv2.imshow('canny', cv_image)
    # cv2.imshow('mask_canny', cpy_img)

    # cv2.imshow('eroded', eroded)
    # cv2.imshow('')

    cv2.imshow('canny mask', cpy_img)


    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(waitTime) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()