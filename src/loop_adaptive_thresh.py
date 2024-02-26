import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

def kernelx(x):
    """
    Returns a square kernel of size x by x.
    """
    return np.ones((x,x),np.uint8)

def gaussian_threshold(img, blockSize, constant):
    """
    Returns an image thresholded using adaptive gaussian thresholding.
    """
    return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,blockSize,constant)

def crop_image(img, crop_top):
    """
    Returns an image cropped from the top.
    """
    rows, cols = img.shape[:2]
    return img[crop_top:rows, 0:cols]


blockSizeGaus = 109
constantGaus = -29
closing_iterations = 1
kernel_size = 3

waitTime = 0
img_count = 0
crop_top = 500

while(1):
    if img_count > 116:
        img_count = 0
    if img_count < 0:
        img_count = 116
    
    img_normal = cv2.imread(f'imgs/img_{img_count}.jpg')
    img = cv2.cvtColor(img_normal, cv2.COLOR_BGR2GRAY)
    #img = cv2.medianBlur(img,5)

    # Crop image to reduce value range and remove sky/background
    cropped_image = crop_image(img, crop_top)
    cv2.imshow("cropped", cropped_image)

    # Gaussian Thresholding
    gaussian = gaussian_threshold(cropped_image, blockSizeGaus, constantGaus)
    
    opening = cv2.morphologyEx(gaussian,cv2.MORPH_OPEN,kernelx(kernel_size), iterations = closing_iterations)
    openclose = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernelx(kernel_size), iterations = closing_iterations)


    linesP2 = cv2.HoughLinesP(openclose, 1, np.pi / 180, 50, None, minLineLength=60, maxLineGap=40)
    lines = cv2.cvtColor(np.zeros_like(openclose), cv2.COLOR_GRAY2BGR)
    if linesP2 is not None:
        for i in range(0, len(linesP2)):
            l = linesP2[i][0]
            cv2.line(lines, (l[0], l[1]), (l[2], l[3]), (255,255,255), 3, cv2.LINE_AA)

    special_kernel = np.array([[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]], np.uint8)

    lines = cv2.threshold(lines, 127, 255, cv2.THRESH_BINARY)[1]
    lines_or_openclose = cv2.bitwise_or(lines, openclose)
    lines = cv2.dilate(lines, special_kernel, iterations = 1)
    lines_dilated = cv2.bitwise_or(lines, openclose)
    cv2.imshow("OpenCLoselines", lines)

    open_open = cv2.morphologyEx(lines_dilated, cv2.MORPH_OPEN, kernelx(3), iterations = 2)
    open_open = cv2.cvtColor(open_open, cv2.COLOR_BGR2GRAY)
    cv2.imshow("OpenOpen", open_open)

    sobel1 = cv2.Sobel(open_open, cv2.CV_8UC1, 1, 0, ksize=3)

    contours_open_open,_ = cv2.findContours(open_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cpy_img = cropped_image.copy()

    if len(contours_open_open) >= 1:
        cv2.drawContours(cropped_image, contours_open_open, -1, (0,255,0), 5)
        for contour in contours_open_open:
            print(cv2.contourArea(contour))        
            x,y,w,h = cv2.boundingRect(contour)
            print("X: " , x , "\tY: ", y, "\tW: " , w ,"\tH: ", h)
            cv2.rectangle(cropped_image, (x,y), (x+w,y+h), (0,0,255), 1)

            centroid, dimensions, angle = cv2.minAreaRect(contour)
            # draw rotated rect
            # rect = cv2.minAreaRect(contour)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # cv2.drawContours(cpy_img,[box],0,(0,0,255),2)


    cv2.imshow('Original Image', img)
    cv2.imshow("Cropped Image", cropped_image)
    cv2.imshow('Adaptive Gaussian Thresholding', gaussian)
    cv2.imshow('Opening', opening)
    cv2.imshow('OpenClose', openclose)
    cv2.imshow('Sobel', sobel1)
    # cv2.imshow('rotated rect', cpy_img)

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
        # print("*" * 24)
        # print("blockSizeGaus: ", blockSizeGaus)
        # print("blockSizeMean: ", blockSizeMean)

    elif cv2.waitKey(waitTime) & 0xFF == ord('2'):
        blockSizeGaus -= 2
        # print("*" * 24)
        # print("blockSizeGaus: ", blockSizeGaus)
        # print("blockSizeMean: ", blockSizeMean)
    elif cv2.waitKey(waitTime) & 0xFF == ord('3'):
        constantGaus += 1
    elif cv2.waitKey(waitTime) & 0xFF == ord('4'):
        constantGaus -= 1
    elif cv2.waitKey(waitTime) & 0xFF == ord('5'):
        closing_iterations += 1
    elif cv2.waitKey(waitTime) & 0xFF == ord('6'):
        closing_iterations -= 1
    elif cv2.waitKey(waitTime) & 0xFF == ord('7'):
        kernel_size += 1
    elif cv2.waitKey(waitTime) & 0xFF == ord('8'):
        kernel_size -= 1
    elif cv2.waitKey(waitTime) & 0xFF == ord('9'):
        crop_top += 5
    elif cv2.waitKey(waitTime) & 0xFF == ord('0'):
        crop_top -= 5
    print("blocksize: ", blockSizeGaus, "\tconstant: ", constantGaus, "\tclosing iterations: ", closing_iterations, "\tkernel size: ", kernel_size, "\theight: ", crop_top)


cv2.destroyAllWindows()