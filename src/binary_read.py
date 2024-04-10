import cv2
import numpy as np
# File paths
file1_path = 'blistener.txt'
file2_path = 'btalk.txt'

# Read binary files
with open(file1_path, 'rb') as file1, open(file2_path, 'rb') as file2:
    file1_data = file1.read()
    file2_data = file2.read()

# Convert binary data to numpy arrays
file1_array = np.frombuffer(file1_data, np.uint8)
file2_array = np.frombuffer(file2_data, np.uint8)

# Create OpenCV images
image1 = cv2.imdecode(file1_array, cv2.IMREAD_COLOR)
image2 = cv2.imdecode(file2_array, cv2.IMREAD_COLOR)

# Display images
cv2.imshow('Listener 1', image1)
cv2.imshow('Talker 2', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()