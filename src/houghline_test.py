import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
def nothing(x):
    pass

def kernelx(x):
    return np.ones((x,x),np.uint8)

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

blockSizeGaus = 61
blockSizeMean = 31
constantGaus = -20
constantMean = -25

output = img
waitTime = 0
height = 1
img_count = 0
while(1):
    if img_count > 116:
        img_count = 0
    file_name = "imgs/img_" + str(img_count) + ".jpg"
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
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
    
    img = cv2.GaussianBlur(img,(5,5),0)
    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    print(blockSizeMean)
    print(blockSizeGaus)

    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,blockSizeGaus,constantGaus)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,blockSizeMean,constantMean)
    
    blank_mask = np.zeros_like(img)
    ignore_mask_color = (255,255,255)
    rows, cols = img.shape[:2]
    cv2.rectangle(blank_mask, (cols, rows), (0, 500), 255, -1)

    masked_image = cv2.bitwise_and(th3, th3, mask = blank_mask)

    closing = cv2.morphologyEx(masked_image, cv2.MORPH_CLOSE, kernelx(9), iterations = 2)
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernelx(7))
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernelx(5))

    #masked_image = cv2.morphologyEx(masked_image,cv2.MORPH_CLOSE,kernel)
    #masked_image = cv2.morphologyEx(masked_image,cv2.MORPH_CLOSE,kernel)
    #closing = cv2.morphologyEx(th3,cv2.MORPH_CLOSE,kernel)
    #masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    cny1 = cv2.Canny(masked_image, 50, 200, None, 3)
    cdstP = cv2.cvtColor(cny1, cv2.COLOR_GRAY2BGR)
    sobel1 = cv2.Sobel(closing, cv2.CV_8UC1, 1, 0, ksize=3)
    sobel1 = cv2.dilate(sobel1,kernelx(3),iterations = 1)
    
    linesP = cv2.HoughLinesP(sobel1, 1, np.pi / 180, 50, None, 80, 50)
    linesIMG = cv2.cvtColor(sobel1, cv2.COLOR_GRAY2BGR)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(linesIMG, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", linesIMG)
    cv2.imwrite("HoughLinesP.jpg", linesIMG)
    cv2.imwrite("base.jpg", img_normal)


    cv2.imshow('Original Image', img)
    # cv2.imshow('Global Thresholding (v = 127)', th1)
    cv2.imshow('Adaptive Mean Thresholding', th2)
    cv2.imshow('Adaptive Gaussian Thresholding', th3)
    cv2.imshow('Closing', closing)
    cv2.imshow('Masked', masked_image)
    cv2.imshow('Sobel', sobel1)


    cv2.imshow('MASK', blank_mask)
    cv2.imshow("img_normal", img_normal)


    # cv2.imshow('guassian contours', cpy_img)
    # cv2.imshow('guassian contours_all', cpy_img_all)
    # cv2.imshow('invert', invert)
    # cv2.imshow('closing invert', closing_invert)
    # cv2.imshow('erosion invert', erosion_invert)


    # cv2.imshow('contours invert', cpy_contours_invert)


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
        print("*" * 24)
        print("blockSizeGaus: ", blockSizeGaus)
        print("blockSizeMean: ", blockSizeMean)
    elif cv2.waitKey(waitTime) & 0xFF == ord('3'):
        constantGaus += 1
        constantMean += 1
    elif cv2.waitKey(waitTime) & 0xFF == ord('4'):
        constantGaus -= 1
        constantMean -= 1
    print("blocksize: ", blockSizeGaus, "\tconstant: ", constantGaus)
        

cv2.destroyAllWindows()