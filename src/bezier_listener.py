import rpyc
import time
import cv2
if __name__ == "__main__":
    c = rpyc.connect("localhost", 9001)
    print("Listener Connected")
    while True:
        img, curve = c.root.get_bezier()
        if img is not None:
            print("Got image")
            # do something with the image
            cv2.imshow('Image', img)
        if curve is not None:
            print("Got curve")
            # do something with the curve
            print(curve)
        time.sleep(0.1)
