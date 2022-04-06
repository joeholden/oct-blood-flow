import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
from flux_module import return_pixels_of_roi, draw_flow, draw_hsv, get_pixels_inside_roi, \
    get_intensity_spectrum_along_path, sum_pixel_intensities_return_peaks, generate_multi_page_tiff_with_guesses

IMG_PATH = 'wt1 test/WT.tif'
IMG_PATH_AVI = 'wt1 test/WT.avi'
FLUX_WINDOW_ROI_PATH = 'wt1 test/window.roi'
FLOW_ROI = 'wt1 test/flow.roi'
RESIZE_IMG_DIMS = (1024, 1024)
PAGE_MIN = 0
PAGE_MAX = 150

# Generate multi-page .tif with colored guesses for cell presence and print flux numbers
px = sum_pixel_intensities_return_peaks(image_path=IMG_PATH, roi_path=FLUX_WINDOW_ROI_PATH,
                                        page_min=PAGE_MIN, page_max=PAGE_MAX)
generate_multi_page_tiff_with_guesses(image_path=IMG_PATH, roi_path=FLUX_WINDOW_ROI_PATH, peak_indices=px)

# Optic Flow Process
cap = cv2.VideoCapture(IMG_PATH_AVI)
process_success, previous_frame = cap.read()
previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
px_in_roi = get_pixels_inside_roi(image_path=IMG_PATH, roi_path=FLUX_WINDOW_ROI_PATH, page=0)

flow_magnitudes = []
for i in range(PAGE_MIN, PAGE_MAX):
    process_success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(previous_gray, gray, None, 0.5, 3, 15, 3, 7, 1.5, 0)
    previous_gray = gray

    flow_vector_image = cv2.resize(draw_flow(gray, flow), RESIZE_IMG_DIMS)
    flow_hsv_image = cv2.resize(draw_hsv(flow), RESIZE_IMG_DIMS)

    cv2.imshow('Flow Vector', flow_vector_image)
    cv2.imshow('Flow HSV', flow_hsv_image)
    cv2.imwrite(f'fv/{i}.tif', flow_vector_image)

    key = cv2.waitKey(5)
    if key == ord('q'):
        break

    magnitude = 0
    for px in px_in_roi:
        flow_mag = math.sqrt(flow[px[0], px[1]][0] ** 2 + flow[px[0], px[1]][0] ** 2)
        magnitude += flow_mag
    flow_magnitudes.append(magnitude)

cap.release()
cv2.destroyAllWindows()

flow_x_array = np.arange(PAGE_MIN, PAGE_MAX, 1)
smooth_flow_mag = scipy.ndimage.gaussian_filter1d(flow_magnitudes, sigma=0.6)
plt.plot(flow_x_array, flow_magnitudes, 'green')
plt.plot(flow_x_array, smooth_flow_mag, 'orange')
median_flow = np.median(flow_magnitudes)

for j in range(PAGE_MIN, PAGE_MAX):
    plt.axvline(x=j)
plt.axhline(median_flow)
plt.show()

