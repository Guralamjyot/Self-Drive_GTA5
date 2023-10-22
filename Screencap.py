import time
import cv2
import mss
import numpy as np

def screen_grab():
    with mss.mss() as sct:
        # Part of the screen to capture
        monitor = {"top": 40, "left": 0, "width": 800, "height": 580}

        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(monitor))


        #print("fps: {}".format(1 / (time.time() - last_time)))
        return(img[:,:,:3])

