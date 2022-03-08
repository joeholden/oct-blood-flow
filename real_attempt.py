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


IMAGE_PATH = 'practice image.tif'
IMG = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
ROI = ImagejRoi.fromfile('path.roi')


def draw_roi(image, imagej_roi, color):
    """Takes in a CV2 image, imageJ ROI, and a bracketed BGR color value [B, G, R]. Saves overlay image and
    returns the pixel array of the roi"""
    img = image
    roi = imagej_roi
    c = color

    # Get ROI coordinates
    top = roi.top
    left = roi.left
    integer_coordinates = roi.integer_coordinates
    integer_coordinates = [list(px) for px in integer_coordinates]
    map_coordinates = [(j + top, i + left) for [i, j] in integer_coordinates]

    # Get Bresenham pixel arrays between each point in curve and add to a single list
    length = len(roi.subpixel_coordinates)
    roi_pixel_array = []
    for i in range(length - 1):
        x1 = map_coordinates[i][0]
        y1 = map_coordinates[i][1]
        x2 = map_coordinates[i + 1][0]
        y2 = map_coordinates[i + 1][1]
        pixel_array_step = list(bresenham(int(x1), int(y1), int(x2), int(y2)))
        roi_pixel_array += pixel_array_step

    for pixel in roi_pixel_array:
        img[pixel] = c

    # cv2.imshow('title', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(f'roi_traced_on_image_{IMAGE_PATH.split(".")[0]}.tif', img)
    return roi_pixel_array


def get_intensity_spectrum(roi_pixels):
    """Uses the ImageJ ROI to calculate an accurate path distance and 8 bit intensity along that path.
    Smooths curve and plots it"""

    def get_distance(x1, y1, x2, y2):
        dst = math.sqrt(abs(x2-x1)**2 + abs(y2-y1)**2)
        return dst

    distance_array = [0]
    length = len(roi_pixels)

    for i in range(length - 1):
        x1 = roi_pixels[i][0]
        y1 = roi_pixels[i][1]
        x2 = roi_pixels[i + 1][0]
        y2 = roi_pixels[i + 1][1]
        d = get_distance(x1=x1, y1=y1, x2=x2, y2=y2)
        distance_array.append(d + distance_array[-1])

    clean_image_copy = cv2.imread(IMAGE_PATH)
    intensity_array = []
    for p in roi_pixels:
        pixel_value = clean_image_copy[p]
        intensity = round(math.sqrt(pixel_value[0]**2 + pixel_value[1]**2),0)
        intensity_array.append(intensity)

    # Savgol params: array, window length, polynomial order
    smooth_intensity = scipy.signal.savgol_filter(intensity_array, 15, 3)

    # Getting Relative Minima
    thresh = 40
    peaks, _ = find_peaks(smooth_intensity * -1)
    plt.scatter(peaks, smooth_intensity[peaks])
    plt.scatter(distance_array, smooth_intensity)
    # Plotting
    # plt.scatter(minima_distance, minima, color='green')
    plt.plot(distance_array, smooth_intensity, color='purple')
    plt.xlabel('Distance in Pixels')
    plt.ylabel('8 bit Intensity')
    plt.title('Intensity Plot')
    plt.show()


roi_px = draw_roi(image=IMG, imagej_roi=ROI, color=[128, 0, 128])
get_intensity_spectrum(roi_px)


