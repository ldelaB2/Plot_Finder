import cv2 as cv
import numpy as np
import multiprocessing as mp
import time
from scipy.interpolate import LSQUnivariateSpline

from functions.display import dialate_skel, save_results
from functions.general import find_consecutive_in_range
from functions.image_processing import find_correct_sized_obj
from functions.general import bindvec

def trim_signal(signal, logger):
    mean_signal = np.mean(signal[signal != 0]).astype(int)
    lower_bound = mean_signal * .5
    upper_bound = mean_signal * 1.5
    min_index = find_consecutive_in_range(signal, lower_bound, upper_bound, 15)
    max_index = find_consecutive_in_range(signal[::-1], lower_bound, upper_bound, 15)
    max_index = len(signal) - max_index

    logger.info(f"Trimming Signal from {min_index} to {max_index}")
    
    return min_index, max_index

def size_filter(img, min_size, logger):
    _, labeled_img, stats, _ = cv.connectedComponentsWithStats(img)
    areas = stats[1:,4]
    remove_areas = np.where(areas < min_size)[0] + 1
    if remove_areas.size == 0:
        return img
    else:
        for e in remove_areas:
            img[labeled_img == e] = 0
        
        logger.info(f"Removing {remove_areas.size} Objects")
        return img

def find_center_line(img, poly_degree, direction, num_cores, logger):
    if direction == "range":
        direction = 1
    elif direction == "row":
        direction = 0
    else:
        logger.error("Invalid direction for find_center_line")
        exit()

    num_obj, labeled_img, _, _ = cv.connectedComponentsWithStats(img)
    x = np.arange(0, img.shape[direction], 1)
    
    with mp.Pool(num_cores) as pool:
        args = [(np.column_stack(np.where(labeled_img == e)), 5, poly_degree, direction, x) for e in range(1, num_obj)]
        results = pool.map(center_line_worker, args)

    skel = np.zeros_like(img)

    for result in results:
        clipped_result = np.clip(result, 0, (img.shape[1 - direction] - 1))
        if direction == 0:
            skel[x, clipped_result] = 1
        else:
            skel[clipped_result, x] = 1

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
    
    dialated_skel = dialate_skel(skel, 10)

    num_obj, labeled_img, stats, centroids = cv.connectedComponentsWithStats(dialated_skel)

    center_distance = np.array([])
    for e in range(1, num_obj - 1):
        distance = abs(centroids[e, 1] - centroids[e + 1, 1]).astype(int)
        center_distance = np.append(center_distance, distance)

    median_dist = np.median(center_distance)
    impute_dist = np.round((center_distance / median_dist)).astype(int)

    obj_to_impute = np.where(impute_dist > 1)[0]

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
            step_size = ((bottom_side[:, 0] - top_side[:, 0]) / impute_dist[obj_to_impute[e]]).astype(int)

            for k in range(impute_dist[obj_to_impute[e]] - 1):
                new_obj = np.copy(top_side)
                new_obj[:, 0] = top_side[:, 0] + step_size * (k + 1)
                skel[new_obj[:, 0], new_obj[:, 1]] = 1
    
    if direction == 1:
        skel = skel.T

    return skel

def process_range_wavepad(wavepad, params, logger):
    num_cores = params["num_cores"]
    box_radi = params["box_radi"]

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

    # Apply the mask
    fft_wavepad = fft_wavepad * final_mask[:, np.newaxis]

    # Find the mean amplitude of the signal across the range
    mean_col_amp = np.mean(np.abs(fft_wavepad), axis = 0)

    # Convert to complex wave
    fft_wavepad = np.abs(fft_wavepad) * np.exp(1j * np.angle(fft_wavepad))

    # Inverse FFT
    fft_wavepad = np.fft.irfft(np.fft.ifftshift(fft_wavepad, axes = 0), axis = 0)

    # Bindvec and convert to uint8
    fft_wavepad = np.round(255 * bindvec(np.real(fft_wavepad))).astype(np.uint8)

    logger.info(f"FFT total time (s): {np.round(time.time() - start_time, 2)}")

    # Save the results
    save_results(params, [fft_wavepad], ["range_wavepad_fft"], "image", logger)

    # Trim the edges
    row_start = box_radi[0]
    row_end = wavepad.shape[0] - box_radi[0]
    fft_wavepad[:row_start, :] = 0
    fft_wavepad[row_end:, :] = 0

    col_start = box_radi[1]
    col_end = wavepad.shape[1] - box_radi[1]
    min_indx, max_indx = trim_signal(mean_col_amp, logger)

    col_trim_start = max(col_start, min_indx)
    col_trim_end = min(col_end, max_indx)

    fft_wavepad[:, :col_trim_start] = 0
    fft_wavepad[:, col_trim_end:] = 0

    # Binary threshold
    _, binary_wavepad = cv.threshold(fft_wavepad, 0, 255, cv.THRESH_OTSU)

    # Find the correct sized objects
    min_obj_size = params["min_obj_size_range"]
    logger.info(f"Filtering Range Wavepad with Minimum Object Size: {min_obj_size}")
    size_filtered_wavepad = size_filter(binary_wavepad, min_obj_size, logger)
    correct_sized_wavepad, avg_obj_area = find_correct_sized_obj(size_filtered_wavepad)
    logger.info(f"Average Object Area: {np.round(avg_obj_area, 2)} for range wavepad")

    # Save the results
    save_results(params, [correct_sized_wavepad], ["range_wavepad_filtered"], "wavepad", logger)

    # Find the center lines
    poly_degree = params["poly_deg_range"]
    direction = "range"

    # Find the center line
    center_line = find_center_line(correct_sized_wavepad, poly_degree, direction, num_cores, logger)

    # Impute the skeleton
    imputed_skel = impute_skel(center_line, direction, logger)

    # Report the total time
    logger.info(f"Total time for range wavepad processing (s): {np.round(time.time() - start_time, 2)}")

    # Save the results
    save_results(params, [center_line, imputed_skel], ["range_skel_original", "range_skel_imputed"], "skel", logger)
                 
    return imputed_skel

def process_row_wavepad(wavepad, params, logger):
    num_cores = params["num_cores"]
    box_radi = params["box_radi"]

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

    # Apply the mask
    fft_wavepad = fft_wavepad * final_mask[np.newaxis, :]

    # Get the mean amplitude of the signal across the rows
    mean_row_amp = np.mean(np.abs(fft_wavepad), axis = 1)

    # Convert to complex wave
    fft_wavepad = np.abs(fft_wavepad) * np.exp(1j * np.angle(fft_wavepad))

    # Inverse FFT
    fft_wavepad = np.fft.irfft(np.fft.ifftshift(fft_wavepad, axes = 1), axis = 1)

    # Bindvec and convert to uint8
    fft_wavepad = np.round(255 * bindvec(np.real(fft_wavepad))).astype(np.uint8)

    logger.info(f"FFT total time (s): {np.round(time.time() - start_time, 2)}")

    # Save the results
    save_results(params, [fft_wavepad], ["row_wavepad_fft"], "image", logger)

    # Trim the edges
    row_start = box_radi[0]
    row_end = wavepad.shape[0] - box_radi[0]
    min_indx, max_indx = trim_signal(mean_row_amp, logger)

    row_trim_start = max(row_start, min_indx)
    row_trim_end = min(row_end, max_indx)

    fft_wavepad[:row_trim_start, :] = 0
    fft_wavepad[row_trim_end:, :] = 0

    col_start = box_radi[1]
    col_end = wavepad.shape[1] - box_radi[1]
    fft_wavepad[:, :col_start] = 0
    fft_wavepad[:, col_end:] = 0

    # Binary threshold
    _, binary_wavepad = cv.threshold(fft_wavepad, 0, 1, cv.THRESH_OTSU)

    # Find the correct sized objects
    min_obj_size = params["min_obj_size_row"]
    logger.info(f"Filtering Row Wavepad with Minimum Object Size: {min_obj_size}")
    size_filtered_wavepad = size_filter(binary_wavepad, min_obj_size, logger)
    correct_sized_wavepad, avg_obj_area = find_correct_sized_obj(size_filtered_wavepad)
    logger.info(f"Average Object Area: {np.round(avg_obj_area, 2)} for row wavepad")

    # Save the results
    save_results(params, [correct_sized_wavepad], ["row_wavepad_filtered"], "wavepad", logger)

    # Find the center lines
    poly_degree = params["poly_deg_row"]
    direction = "row"

    # Find the center line
    center_line = find_center_line(correct_sized_wavepad, poly_degree, direction, num_cores, logger)

    # Impute the skeleton
    imputed_skel = impute_skel(center_line, direction, logger)

    logger.info(f"Total time for row wavepad processing (s): {np.round(time.time() - start_time, 2)}")

    # Save the results
    save_results(params, [center_line, imputed_skel], ["row_skel_original", "row_skel_imputed"], "skel", logger)

    return imputed_skel


def wavepad_worker(signal):
    signal = signal - np.mean(signal)
    fft_signal = np.fft.rfft(signal)
    fft_signal = np.fft.fftshift(fft_signal)
    return fft_signal

def center_line_worker(args):
    points, n_knots, poly_degree, direction, x = args

    original_y = points[:, 1 - direction]
    min_y = np.min(original_y)
    max_y = np.max(original_y)

    unique_x, indices = np.unique(points[:, direction], return_inverse = True)
    y_sum = np.zeros_like(unique_x)
    np.add.at(y_sum, indices, points[:, (1 - direction)])
    mean_values = (y_sum / np.bincount(indices)).astype(int)

    knots = np.linspace(unique_x[0], unique_x[-1], n_knots)[1:-1]
    spl = LSQUnivariateSpline(unique_x, mean_values, knots, k = poly_degree)
    y = spl(x).astype(int)

    y = np.clip(y, min_y, max_y)

    return y





    