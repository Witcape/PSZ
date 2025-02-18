import cv2
import numpy as np

def airlight(HazeImg, wsz):
    A = []
    for k in range(3):
        minImg = cv2.erode(HazeImg[:, :, k].astype(np.float64), np.ones((wsz, wsz)), borderType=cv2.BORDER_REFLECT)
        A.append(np.max(minImg))
    return np.array(A)