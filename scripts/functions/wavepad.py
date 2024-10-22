import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

from scipy.interpolate import LSQUnivariateSpline

from functions.general import find_consecutive_in_range
from functions.image_processing import find_correct_sized_obj
from functions.general import bindvec
import multiprocessing as mp
import time


def trim_boarder(img, direction, logger):
    if direction == "range":
        direction = 0
    elif direction == "row":
        direction = 1
    else:
        logger.error("Invalid Trim Boarder Direction")
        exit()

    t_sum = np.sum(img, axis = direction)
    t_mean = np.mean(t_sum[t_sum != 0]).astype(int)
    lower_bound = t_mean - 50
    upper_bound = t_mean + 50
    min_index = find_consecutive_in_range(t_sum, lower_bound, upper_bound, 15)
    max_index = find_consecutive_in_range(t_sum[::-1], lower_bound, upper_bound, 15)
    max_index = t_sum.size - max_index

    logger.info(f"Trimming Boarder: {min_index} to {max_index} in {direction} direction")
    
    if direction == 0:
        img[:, :(min_index + 1)] = 0
        img[:, max_index:] = 0
    else:
        img[:(min_index + 1), :] = 0
        img[max_index:, :] = 0
    
    return img

def find_center_line(img, poly_degree, direction, logger):
    if direction == "range":
        direction = 1
    elif direction == "row":
        direction = 0
    else:
        logger.error("Invalid direction for find_center_line")
        exit()

    num_obj, labeled_img, _, _ = cv.connectedComponentsWithStats(img)
    skel = np.zeros_like(img)

    for e in range(1, num_obj):
        subset = np.column_stack(np.where(labeled_img == e))
        unique_values, counts = np.unique(subset[:, direction], return_counts=True)
        position = np.bincount(subset[:, direction], weights=subset[:, (1 - direction)])
        mean_values = (position[unique_values] / counts).astype(int)
        n_knots = 5
        knots = np.linspace(unique_values[0], unique_values[-1], n_knots)[1:-1]
        spl = LSQUnivariateSpline(unique_values, mean_values, knots, k = poly_degree)
        x = np.arange(0, img.shape[direction], 1)
        y = spl(x).astype(int)

        if direction == 0:
            img_points = skel[x,y]
            if np.max(img_points) == 0:
                skel[x, y] = 1
        else:
            img_points = skel[y,x]
            if np.max(img_points) == 0:
                skel[y, x] = 1

    return skel

def impute_skel(skel, direction, logger):
    if direction == "range":
        direction = 0
    elif direction == "row":
        direction = 1
    else:
        logger.error("Invalid direction for impute_skel")
        exit()

    if direction == 1:
        skel = skel.T
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    dialated_skel = cv.dilate(skel, kernel)

    num_obj, labeled_img, stats, centroids = cv.connectedComponentsWithStats(dialated_skel)

    center_distance = np.array([])
    for e in range(1, num_obj - 1):
        distance = abs(centroids[e, 1] - centroids[e + 1, 1]).astype(int)
        center_distance = np.append(center_distance, distance)

    median_dist = np.median(center_distance)
    avg_dist = (median_dist * 1.15).astype(int)

    obj_to_impute = np.where(center_distance > avg_dist)[0]
    impute_dist = np.round((center_distance[obj_to_impute] / median_dist)).astype(int)

    if obj_to_impute.size == 0:
        logger.info("No Objects to Impute")
    else:
        logger.info(f"Imputing {obj_to_impute.size} Objects")

        indx = np.where(skel != 0)
        skel[indx] = labeled_img[indx]

        for e in range(obj_to_impute.size):
            top_indx = obj_to_impute[e] + 1
            bottom_indx = obj_to_impute[e] + 2
            top_side = np.column_stack(np.where(skel == top_indx))
            bottom_side = np.column_stack(np.where(skel == bottom_indx))
            top_side = top_side[np.argsort(top_side[:, 1])]
            bottom_side = bottom_side[np.argsort(bottom_side[:, 1])]
            step_size = ((bottom_side[:, 0] - top_side[:, 0]) / impute_dist[e]).astype(int)

            for k in range(impute_dist[e] - 1):
                new_obj = np.copy(top_side)
                new_obj[:, 0] = top_side[:, 0] + step_size * (k + 1)
                skel[new_obj[:, 0], new_obj[:, 1]] = 100

    skel[skel != 0] = 1
    skel = skel.astype(np.uint8)
    
    if direction == 1:
        skel = skel.T

    return skel

def process_range_wavepad(wavepad, params, logger):
    num_cores = params["num_cores"]

    # Invert the wavepad
    wavepad = 255 - wavepad

    logger.info("Starting to Processing Range wavepad")
    start_time = time.time()
    with mp.Pool(num_cores) as pool:
        fft_results = pool.map(wavepad_worker, [wavepad[:, i] for i in range(wavepad.shape[1])])

    logger.info(f"FFT Generation Time (s): {np.round(time.time() - start_time, 2)}")

    fft_wavepad = np.column_stack(fft_results)

    # Compute the expected signal in pixels
    gsd = params["GSD"]
    range_spacing_cm = params["implied_range_spacing_ft"] * 30.48
    pixel_signal = np.round(range_spacing_cm / gsd).astype(int)
    signal_freq = np.round(wavepad.shape[0] / pixel_signal).astype(int)
    
    # Find the mean amplitude of the signal
    mean_amp = np.mean(np.abs(fft_wavepad), axis = 1)

    # Compute the frequency bounds
    lower_bound = (signal_freq * .75).astype(int)
    upper_bound = (signal_freq * 1.25).astype(int)

    # Create the first frequency mask
    freq_mask = np.zeros_like(mean_amp)
    center = len(mean_amp) // 2
    freq_mask[(center + lower_bound):(center + upper_bound)] = 1

    # Filter the amplitude    
    filtered_amp = mean_amp * freq_mask

    # Get the two largest signals
    signal = np.argsort(filtered_amp)[::-1][:1]
    
    # Create the final signal mask
    final_mask = np.zeros_like(mean_amp)
    final_mask[signal] = 1

    def range_filter(fft_signal):
        filtered_amp = np.abs(fft_signal) * final_mask
        filtered_angle = np.angle(fft_signal) * final_mask
        spacial_wave = filtered_amp * np.exp(1j * filtered_angle)
        spacial_wave = np.real(np.fft.irfft(np.fft.ifftshift(spacial_wave)))
        spacial_wave = np.round(spacial_wave).astype(np.int16)
        return spacial_wave
    
    fft_wavepad = np.apply_along_axis(range_filter, 0, fft_wavepad)
    fft_wavepad = np.round(255 * bindvec(fft_wavepad)).astype(np.uint8)

    logger.info(f"FFT total time (s): {np.round(time.time() - start_time, 2)}")

    # Binary threshold
    _, binary_wavepad = cv.threshold(fft_wavepad, 0, 1, cv.THRESH_OTSU)

    ones = np.sum(np.where(binary_wavepad == 1))
    zeros = np.sum(np.where(binary_wavepad == 0))

    # Invert if needed
    if ones > zeros:
        binary_wavepad = 1 - binary_wavepad

    # Find the correct sized objects
    min_obj_size = params["min_obj_size_range"]
    poly_degree = params["poly_deg_range"]
    direction = "range"

    # Trim the boarder
    trimmed_wavepad = trim_boarder(binary_wavepad, direction, logger)

     # Find correct sized objects
    logger.info(f"Finding Correct Sized Objects with min size: {min_obj_size}")
    correct_sized_wavepad, avg_obj_area = find_correct_sized_obj(trimmed_wavepad, min_obj_size)
    logger.info(f"Average Object Area: {np.round(avg_obj_area, 2)} for {direction} wavepad")

    # Find the center line
    center_line = find_center_line(correct_sized_wavepad, poly_degree, direction, logger)

    # Impute the skeleton
    imputed_skel = impute_skel(center_line, direction, logger)

    logger.info(f"Total time for range wavepad processing (s): {np.round(time.time() - start_time, 2)}")

    return imputed_skel

def process_row_wavepad(wavepad, params, logger):
    num_cores = params["num_cores"]

    # Invert the wavepad
    wavepad = 255 - wavepad

    logger.info("Starting to Processing Row wavepad")
    start_time = time.time()
    with mp.Pool(num_cores) as pool:
        fft_results = pool.map(wavepad_worker, [wavepad[i, :] for i in range(wavepad.shape[0])])

    logger.info(f"FFT Generation Time (s): {np.round(time.time() - start_time, 2)}")

    fft_wavepad = np.row_stack(fft_results)

    # Compute the expected signal in pixels
    gsd = params["GSD"]
    range_spacing_cm = params["implied_row_spacing_in"] * 2.54
    pixel_signal = np.round(range_spacing_cm / gsd).astype(int)
    signal_freq = np.round(wavepad.shape[1] / pixel_signal).astype(int)
    
    # Find the mean amplitude of the signal
    mean_amp = np.mean(np.abs(fft_wavepad), axis = 0)

    # Compute the frequency bounds
    lower_bound = (signal_freq * .75).astype(int)
    upper_bound = (signal_freq * 1.25).astype(int)

    # Create the first frequency mask
    freq_mask = np.zeros_like(mean_amp)
    center = len(mean_amp) // 2
    freq_mask[(center + lower_bound):(center + upper_bound)] = 1

    # Filter the amplitude    
    filtered_amp = mean_amp * freq_mask

    # Get the two largest signals
    signal = np.argsort(filtered_amp)[::-1][:1]

    # Create the final signal mask
    final_mask = np.zeros_like(mean_amp)
    final_mask[signal] = 1

    def row_filter(fft_signal):
        filtered_amp = np.abs(fft_signal) * final_mask
        filtered_angle = np.angle(fft_signal) * final_mask
        spacial_wave = filtered_amp * np.exp(1j * filtered_angle)
        spacial_wave = np.real(np.fft.irfft(np.fft.ifftshift(spacial_wave)))
        spacial_wave = np.round(spacial_wave).astype(np.int16)
        return spacial_wave
    
    fft_wavepad = np.apply_along_axis(row_filter, 1, fft_wavepad)
    fft_wavepad = np.round(255 * bindvec(fft_wavepad)).astype(np.uint8)

    logger.info(f"FFT total time (s): {np.round(time.time() - start_time, 2)}")

    # Binary threshold
    _, binary_wavepad = cv.threshold(fft_wavepad, 0, 1, cv.THRESH_OTSU)

    ones = np.sum(np.where(binary_wavepad == 1))
    zeros = np.sum(np.where(binary_wavepad == 0))

    # Invert if needed
    if ones > zeros:
        binary_wavepad = 1 - binary_wavepad

    # Find the correct sized objects
    min_obj_size = params["min_obj_size_row"]
    poly_degree = params["poly_deg_row"]
    direction = "row"

    # Trim the boarder
    trimmed_wavepad = trim_boarder(binary_wavepad, direction, logger)

    # Find correct sized objects
    logger.info(f"Finding Correct Sized Objects with min size: {min_obj_size}")
    correct_sized_wavepad, avg_obj_area = find_correct_sized_obj(trimmed_wavepad, min_obj_size)
    logger.info(f"Average Object Area: {np.round(avg_obj_area, 2)} for {direction} wavepad")

    # Find the center line
    center_line = find_center_line(correct_sized_wavepad, poly_degree, direction, logger)

    # Impute the skeleton
    imputed_skel = impute_skel(center_line, direction, logger)

    logger.info(f"Total time for row wavepad processing (s): {np.round(time.time() - start_time, 2)}")

    return imputed_skel


def wavepad_worker(signal):
    signal = signal - np.mean(signal)
    fft_signal = np.fft.rfft(signal)
    fft_signal = np.fft.fftshift(fft_signal)
    return fft_signal


    