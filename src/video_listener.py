import rpyc
import time
import cv2
if __name__ == "__main__":
    c = rpyc.connect("localhost", 9001)
    print("Listener Connected")
    while True:
        img = c.root.get_image()
        if img is not None:
            print(img)
            # cv2.imshow("sent image", img)
        time.sleep(2)