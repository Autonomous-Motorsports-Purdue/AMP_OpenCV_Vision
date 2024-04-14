import rpyc
import time
import cv2
import numpy as np
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
if __name__ == "__main__":
    c = rpyc.connect("localhost", 9001)
    print("Listener Connected")
    while True:
        # img, curve = c.root.get_bezier()
        img_bytes, curve, control_points = c.root.get_bezier()
        # img2 = np.array(img)
        # img = np.array(img)
        if img_bytes is not None:
            binary_file = open("blistener.txt", "wb")
            binary_file.write(img_bytes)
            binary_file.close()

            # print(img_bytes)

            # print(img)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            np_copy = np_arr.copy()
            img = cv2.imdecode(np_copy, cv2.IMREAD_COLOR)
            # img = np.copy(np.array(img))

            # print(img)

            # cv2.imwrite("talk/img.jpg", img)
            cv2.imshow("img", img)

            # ac_img = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)
            # np.savetxt('test.out', img)
            # print(img)
            # arr = np.loadtxt('test.out')
            # print(ac_img)
            # cv2.imshow("listener img", ac_img)

            # cv2.imshow("listenre img", img)
        if curve is not None:
            print(curve)

        """
        if img2 is not None:
            print("Got image")
            # do something with the image
            print(img)
            cv2.imshow('Image', img2)
        """
        # if curve is not None:
            # print("Got curve")
            # do something with the curve
            # print(curve)
        time.sleep(1.5)
