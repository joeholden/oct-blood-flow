from roifile import ImagejRoi
from bresenham import bresenham
import cv2
import math
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.signal import find_peaks
import numpy as np
import imageio
import itertools as it
import pathlib
import os
import xml.etree.ElementTree as ET


def return_pixels_of_roi(image, image_path, imagej_roi, color, save_image=False):
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

    if save_image:
        cv2.imwrite(f'roi_traced_on_image_{image_path.split(".")[0]}.tif', img)
    return roi_pixel_array


def get_intensity_spectrum_along_path(image_path, roi_pixels):
    """Uses the ImageJ ROI to calculate an accurate path distance and 8 bit intensity along that path.
    Smooths curve and plots it"""

    def get_distance(x1, y1, x2, y2):
        dst = math.sqrt(abs(x2 - x1) ** 2 + abs(y2 - y1) ** 2)
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

    clean_image_copy = cv2.imread(image_path)
    intensity_array = []
    for p in roi_pixels:
        pixel_value = clean_image_copy[p]
        intensity = round(math.sqrt(pixel_value[0] ** 2 + pixel_value[1] ** 2), 0)
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


def get_pixels_inside_roi(image_path, roi_path, page):
    """Returns an array of pixels that are inside the ROI"""
    ret, images = cv2.imreadmulti(image_path)
    img = images[page]

    roi = ImagejRoi.fromfile(roi_path)
    # The below function also burns the roi onto the image
    roi_px = return_pixels_of_roi(image=img, image_path=image_path, imagej_roi=roi, color=[128, 0, 128])

    shape = img.shape
    inside_pixels = []

    for i in range(shape[1]):
        found_pixels = []
        for j in range(shape[0]):
            if img[j, i][0] == 128 and img[j, i][2] == 128 and img[j, i][1] == 0:
                found_pixels.append((j, i))
        for entry in found_pixels:
            inside_pixels.append(entry)
        if len(found_pixels) > 1:
            j_vals = [px[0] for px in found_pixels]
            min_j = min(j_vals)
            max_j = max(j_vals)
            for col in range(min_j, max_j + 1):
                inside_pixels.append((col, i))
    return inside_pixels


def sum_pixel_intensities_return_peaks(image_path, roi_path, page_min, page_max):
    """Takes in the page range for .tif
    For each page in that tif, gets summed intensities of the pixels in the ROI.
    Returns the peaks"""
    inside_pixels = get_pixels_inside_roi(image_path, roi_path, 0)
    ret, images = cv2.imreadmulti(image_path)

    intensity_array = []
    for i in range(page_min, page_max):
        clean_image_copy = images[i]
        # Get summed intensities of all pixels within the ROI
        intensity_sum = 0
        for p in inside_pixels:
            pixel_value = clean_image_copy[p]
            intensity = round(math.sqrt(pixel_value[0] ** 2 + pixel_value[1] ** 2 + pixel_value[2] ** 2), 0)
            intensity_sum += intensity
        intensity_array.append(intensity_sum)

    # Smooth Curve Slightly to eliminate getting two peaks in adjacent images just because of noise
    smooth_y = scipy.ndimage.gaussian_filter1d(intensity_array, sigma=0.6)
    threshold = ((max(smooth_y) - min(smooth_y))/2 + min(smooth_y))
    threshold = np.median(smooth_y)
    # Above threshold = 0, Below threshold = 1
    above_or_below_thresh = [1 if i < threshold else 0 for i in smooth_y]
    zipped_smooth_y = zip(smooth_y, above_or_below_thresh)

    num_groups = 0
    ng_2 = 0
    for key, group in it.groupby(zipped_smooth_y, lambda x: x[-1]):
        if key == 1:
            num_groups += 1
        else:
            ng_2 +=1
    print(num_groups, ng_2)
    # Groupby above or below threshold

    peaks = [(index, value) for index, value in enumerate(smooth_y) if value <= threshold]

    plt.plot(np.arange(0, page_max-page_min, 1), smooth_y, 'purple')
    plt.plot(np.arange(0, page_max-page_min, 1), np.full((page_max-page_min,), threshold), 'green')

    # plt.plot(np.arange(0, 150, 1), intensity_array, 'orange')
    plt.scatter([i for (i, j) in peaks], [j for (i, j) in peaks])
    plt.savefig('plot.png')
    print(len(peaks))
    print(len(peaks) / (page_max - page_min + 1))
    return [i for (i, j) in peaks]


def generate_multi_page_tiff_with_guesses(image_path, roi_path, peak_indices):
    # Generate validation images
    ret, images = cv2.imreadmulti(image_path)
    list_of_images = []

    for i in range(0, 150):
        IMG = images[i]
        clean_image_copy = IMG.copy()
        ROI = ImagejRoi.fromfile(roi_path)
        if i in peak_indices:
            roi_px = return_pixels_of_roi(image=IMG, image_path=image_path, imagej_roi=ROI, color=[0, 250, 0])
        else:
            roi_px = return_pixels_of_roi(image=IMG, image_path=image_path, imagej_roi=ROI, color=[250, 0, 0])
        list_of_images.append(IMG)

    multi_page_tiff_array = np.stack(tuple(list_of_images), axis=0)
    imageio.mimwrite('multi-page_with_guesses.tiff', multi_page_tiff_array)


def draw_flow(img, flow, step=6):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 10, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def convert_image(relative_image_path, channel):
    """Takes in a .tif file saved directly from Thor Labs .RAW. This .tif should have been opened in imageJ as 16-bit
     unsigned, little-endian byte order, with the proper dimensions and number of images. It also takes in the parameter
      channel. The purpose of the function is to separate out the channels and save only the single channel image as a
      multi-page .tif. Channel numeration starts at 0. Returns length of original and final stacks in that order"""

    # 4 Channels, Fluorescein is in channel 2 so position 1
    file_name = relative_image_path.split('.')[0]

    working_dir = pathlib.PureWindowsPath(os.getcwd()).as_posix()
    if not os.path.isdir(working_dir + '/Images to Process'):
        os.mkdir(working_dir + '/Images to Process')
    if not os.path.isdir(working_dir + '/Processed Raw Image Tifs'):
        os.mkdir(working_dir + '/Processed Raw Image Tifs')

    ret, images = cv2.imreadmulti(working_dir + '/Images to Process/' + relative_image_path)
    list_of_images = []
    array = np.arange(channel, len(images), 4)
    for position in array:
        list_of_images.append(images[position])

    multi_page_tiff_array = np.stack(tuple(list_of_images), axis=0)
    imageio.mimwrite(f'Processed Raw Image Tifs/{file_name}.tif', multi_page_tiff_array)
    return len(images), math.floor(len(images) / 4)


def parse_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()

    name = root[0].attrib['name']
    timepoints = None
    pixelX = None
    pixelY = None

    for child in root:
        if child.tag == 'Timelapse':
            timepoints = int(child.attrib['timepoints'])
        if child.tag == 'LSM':
            pixelX = int(child.attrib['pixelX'])
            pixelY = int(child.attrib['pixelY'])

    print(f'Timepoints: {timepoints}\nPixel Dimensions: {pixelX, pixelY}')
    return timepoints, (pixelX, pixelY)

