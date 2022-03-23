from roifile import ImagejRoi
from bresenham import bresenham
import cv2
import math
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
import numpy as np

# Generate validation images
# multi_tiff = np.empty((256, 128, 3))
# print(np.shape(multi_tiff))
# ret, images = cv2.imreadmulti('rgbc.tif')
# for i in range(0, 150):
#     IMG = images[i]
#     if i == 0:
#         multi_tiff = IMG
#     else:
#         np.append((multi_tiff, IMG))
#
#
# print(np.shape(multi_tiff))


a1 = np.empty((2, 2, 2))
a2 = np.array([[[0,0],[0,0]],[[0,0],[0,0]]])
a3 = np.array([[[0,0],[0,0]],[[0,0],[0,0]]])
print(np.shape(a1))
print(np.shape(a2))


a4 = np.stack((a1, a2), axis=0)
a5 = np.stack((a4, a3), axis=0)
print(np.shape(a4))