import cv2
import numpy as np
from math import factorial

def normalize_path_length(points):
    """
    Returns a list of the normalized path length of the points.
    """
    path_length = [0]
    x, y = points[:,0], points[:,1]

    # calculate the path length
    for i in range(1, len(points)):
        path_length.append(np.sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2) + path_length[i - 1])
    
    # normalize the path length
    # computes the percentage of path length at each point
    pct_len = []
    for i in range(len(path_length)):
        if (path_length[i] == 0):
            pct_len.append(0.01)
            continue
        pct_len.append(path_length[i] / path_length[-1])
    
    return pct_len

def get_bezier(points):
    """
    Returns the control points of a bezier curve.
    """
    num_points = len(points)

    x, y = points[:,0], points[:,1]

    # bezier matrix for a cubic curve
    bezier_matrix = np.array([[-1, 3, -3, 1,], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]])
    bezier_inverse = np.linalg.inv(bezier_matrix)

    normalized_length = normalize_path_length(points)

    points_matrix = np.zeros((num_points, 4))

    for i in range(num_points):
        points_matrix[i] = [normalized_length[i]**3, normalized_length[i]**2, normalized_length[i], 1]

    points_transpose = points_matrix.transpose()
    square_points = np.matmul(points_transpose, points_matrix)

    square_inverse = np.zeros_like(square_points)

    if (np.linalg.det(square_points) == 0):
        print("Uninvertible matrix")
        square_inverse = np.linalg.pinv(square_points)
    else:
        square_inverse = np.linalg.inv(square_points)

    # solve for the solution matrix
    solution = np.matmul(np.matmul(bezier_inverse, square_inverse), points_transpose)

    # solve for the control points
    control_points_x = np.matmul(solution, x)
    control_points_y = np.matmul(solution, y)

    return list(zip(control_points_x, control_points_y))

def comb(n, k):
    """
    Returns the combination of n choose k.
    """
    return factorial(n) / factorial(k) / factorial(n - k)

def plot_bezier(t, cp):
    """
    Plots a bezier curve.
    t is the time values for the curve.
    cp is the control points of the curve.
    return is a tuple of the x and y values of the curve.
    """
    cp = np.array(cp)
    num_points, d = np.shape(cp)   # Number of points, Dimension of points
    num_points = num_points - 1
    curve = np.zeros((len(t), d))
    
    for i in range(num_points+1):
        # Bernstein polynomial
        val = comb(num_points,i) * t**i * (1.0-t)**(num_points-i)
        curve += np.outer(val, cp[i])
    
    return curve

def draw_bezier_curve(img, contour, x_shift):
    """
    Draws a bezier curve on the image.
    """
    # choose every 8th point so that the bezier curve is not too complex and it's faster
    contour_points = np.transpose(np.nonzero(contour))[0::8]
    control_points = np.array(get_bezier(contour_points))
    t = np.linspace(0, 1, 40)
    curve = plot_bezier(t, control_points)
    curve = np.flip(curve, axis=1)
    curve[:,0] += x_shift
    cv2.polylines(img, [np.int32(curve)], isClosed=False, color=(255, 255, 255), thickness=2)

    return curve

def find_two_largest_contours(contours):
    """
    Returns the two largest contours in the list of contours.
    """
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    return largest_contours

def crop_to_contour(img, contour):
    """
    Returns an image cropped to the contour.
    """
    x,y,w,h = cv2.boundingRect(contour)
    return img[y:y+h, x:x+w]
    
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

def crop_image_top(img, crop_top):
    """
    Returns an image cropped from the top.
    """
    rows, cols = img.shape[:2]
    return img[crop_top:rows, 0:cols]


blockSizeGaus = 117
constantGaus = -17
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
    cropped_image = crop_image_top(img, crop_top)
    cv2.imshow("cropped", cropped_image)

    # Gaussian Thresholding
    gaussian = gaussian_threshold(cropped_image, blockSizeGaus, constantGaus)
    
    opening = cv2.morphologyEx(gaussian,cv2.MORPH_OPEN,kernelx(kernel_size), iterations = closing_iterations)
    openclose = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernelx(kernel_size), iterations = closing_iterations)


    linesP2 = cv2.HoughLinesP(openclose, 1, np.pi / 180, 50, None, minLineLength=60, maxLineGap=40)
    lines = np.zeros_like(openclose) #cv2.cvtColor(np.zeros_like(openclose), cv2.COLOR_GRAY2BGR)

    if linesP2 is not None:
        for i in range(0, len(linesP2)):
            l = linesP2[i][0]
            cv2.line(lines, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)

    special_kernel = np.array([[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]], np.uint8)

    ret, lines = cv2.threshold(lines, 127, 255, cv2.THRESH_BINARY)
    lines = cv2.dilate(lines, special_kernel, iterations = 1)
    lines_dilated = cv2.bitwise_or(lines, openclose)
    cv2.imshow("OpenCLoselines", lines)
    
    open_open = cv2.morphologyEx(lines_dilated, cv2.MORPH_OPEN, kernelx(3), iterations = 2)
    cv2.imshow("OpenOpen", open_open)

    sobel1 = cv2.Sobel(open_open, cv2.CV_8UC1, 1, 0, ksize=3)

    contours_open_open,_ = cv2.findContours(open_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cpy_img = cropped_image.copy()

    if len(contours_open_open) >= 1:
        cv2.drawContours(cropped_image, contours_open_open, -1, (0,255,0), 5)
        for contour in contours_open_open:
            #print(cv2.contourArea(contour))        
            x,y,w,h = cv2.boundingRect(contour)
            #print("X: " , x , "\tY: ", y, "\tW: " , w ,"\tH: ", h)
            cv2.rectangle(cropped_image, (x,y), (x+w,y+h), (0,0,255), 1)

            centroid, dimensions, angle = cv2.minAreaRect(contour)
            # draw rotated rect
            # rect = cv2.minAreaRect(contour)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # cv2.drawContours(cpy_img,[box],0,(0,0,255),2)

    largest = find_two_largest_contours(contours_open_open)

    contour1 = crop_to_contour(sobel1, largest[0])
    contour2 = crop_to_contour(sobel1, largest[1])

    curves = np.zeros_like(sobel1)
    curve1 = draw_bezier_curve(curves, contour1, cv2.boundingRect(largest[0])[0])
    curve2 = draw_bezier_curve(curves, contour2, cv2.boundingRect(largest[1])[0])
    

    midpoint_line = (curve1 + curve2) / 2

    cv2.polylines(cropped_image, [np.int32(midpoint_line)], isClosed=False, color=(255, 255, 0), thickness=2)
    cv2.imshow('curve', curves)
    cv2.imshow('Contour1', contour1)
    cv2.imshow('Contour2', contour2)


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