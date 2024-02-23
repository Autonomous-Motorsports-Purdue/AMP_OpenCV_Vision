import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

def kernelx(x):
    return np.ones((x,x),np.uint8)

def gaussian_threshold(img, blockSize, constant):
    return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,blockSize,constant)

def crop_image(img, crop_top):
    rows, cols = img.shape[:2]
    return img[crop_top:rows, 0:cols]


blockSizeGaus = 109
constantGaus = -29
closing_iterations = 1
kernel_size = 3
# 55 -22
# 31 -25

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

    gaussian = gaussian_threshold(cropped_image, blockSizeGaus, constantGaus)
    
    opening = cv2.morphologyEx(gaussian,cv2.MORPH_OPEN,kernelx(kernel_size), iterations = closing_iterations)
    closing_many = cv2.morphologyEx(gaussian,cv2.MORPH_CLOSE,kernelx(kernel_size), iterations = closing_iterations)
    openclose = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernelx(kernel_size), iterations = closing_iterations)

    # sobel1 = cv2.Sobel(openclose, cv2.CV_8UC1, 1, 0, ksize=3)
    # sobel1 = cv2.dilate(sobel1,kernelx(5),iterations = 1)
    
    # linesP = cv2.HoughLinesP(sobel1, 1, np.pi / 180, 50, None, minLineLength=50, maxLineGap=20)
    # linesIMG = cv2.cvtColor(sobel1, cv2.COLOR_GRAY2BGR)
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv2.line(linesIMG, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    # cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", linesIMG)

    linesP2 = cv2.HoughLinesP(openclose, 1, np.pi / 180, 50, None, minLineLength=50, maxLineGap=20)
    linesIMG2 = cv2.cvtColor(openclose, cv2.COLOR_GRAY2BGR)
    if linesP2 is not None:
        for i in range(0, len(linesP2)):
            l = linesP2[i][0]
            cv2.line(linesIMG2, (l[0], l[1]), (l[2], l[3]), (255,255,255), 3, cv2.LINE_AA)


    im_bw = cv2.threshold(linesIMG2, 127, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("OpenCLoselines", im_bw)

    open_open = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernelx(3), iterations = 2)

    cv2.imshow("OpenOpen", open_open)

    sobel1 = cv2.Sobel(open_open, cv2.CV_8UC1, 1, 0, ksize=3)

    cv2.imshow('Original Image', img)
    cv2.imshow("Cropped Image", cropped_image)
    cv2.imshow('Adaptive Gaussian Thresholding', gaussian)
    cv2.imshow('Closing Many', closing_many)
    cv2.imshow('Opening', opening)
    cv2.imshow('OpenClose', openclose)
    cv2.imshow('Sobel', sobel1)

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