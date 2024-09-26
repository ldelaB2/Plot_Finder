import cv2 as cv
import numpy as np
from scipy.optimize import dual_annealing
from scipy.stats import poisson
from matplotlib import pyplot as plt

from functions.general import find_consecutive_in_range
from functions.image_processing import find_correct_sized_obj

def hist_filter_wavepad(wavepad):
    def model_function(x, lambda1, lambda2, p):
        left_dist = p * poisson.pmf(x, mu = lambda1) 
        right_dist = (1 - p) * poisson.pmf(255 - x, mu = lambda2)
        total_dist = left_dist + right_dist
        return total_dist
    
    def calc_prob(x, lambda1, lambda2, p):
        left_dist = p * poisson.pmf(x, mu = lambda1) 
        right_dist = (1 - p) * poisson.pmf(255 - x, mu = lambda2)
        total_dist = left_dist + right_dist
        
        left_prob = np.where(total_dist > 0, left_dist / total_dist, 0)
        right_prob = np.where(total_dist > 0, right_dist / total_dist, 0)
        return left_prob, right_prob
    
    def objective_function(params, bin_centers, pixels_hist):
        prob_dist = model_function(bin_centers, *params)
        dist = - np.sum(pixels_hist * np.log(prob_dist + 1e-10))
        return dist
    
    pixels = wavepad.reshape(-1)
    pixels_hist, _ = np.histogram(pixels, bins = 256, density=True)
    bin_left_edge = np.arange(0, 256, 1)

    bounds = [(0,1), (0,1), (0,1)]
    results = dual_annealing(objective_function, bounds, args=(bin_left_edge, pixels_hist), maxiter = 1000)
    opt_params = results.x

    # Compute the probability of each point
    point_left_prob, point_right_prob = calc_prob(pixels, *opt_params)
    binary_wavepad = np.where(point_left_prob >= point_right_prob, 0, 1)

    # Create the wavepad
    binary_wavepad = binary_wavepad.reshape(wavepad.shape)

    return binary_wavepad

def filter_wavepad(wavepad, logger, method = "otsu"):
    if method == "otsu":
        _, binary_wavepad = cv.threshold(wavepad, 0, 1, cv.THRESH_OTSU)
        logger.info("Otsu Thresholding Complete")

    elif method == "hist":
        binary_wavepad = hist_filter_wavepad(wavepad).astype(np.uint8)
        logger.info("Histogram Thresholding Complete")
        
    else:
        logger.error("Invalid Wavepad Filtering Method")
        exit()

    return binary_wavepad

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
        coefficients = np.polyfit(unique_values, mean_values, poly_degree)
        poly = np.poly1d(coefficients)
        x = np.arange(0, img.shape[direction], 1)
        y = poly(x).astype(int)

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

def process_wavepad( wavepad, poly_degree, direction, min_obj_size, closing_iterations, logger):
    # Filter the wavepad
    filtered_wavepad = filter_wavepad(wavepad, logger)

    # Trim the boarder
    trimmed_wavepad = trim_boarder(filtered_wavepad, direction, logger)

    # Close the wavepad
    if direction == "range":
        logger.info(f"Closing Wavepad with {closing_iterations} iterations")
        kernel = np.ones((5,5), np.uint8)
        closed_wavepad = cv.morphologyEx(trimmed_wavepad, cv.MORPH_CLOSE, kernel, iterations = closing_iterations)
    else:
        logger.info(f"Erroding Wavepad with {closing_iterations} iterations")
        kernel = np.ones((3,3), np.uint8)
        closed_wavepad = cv.morphologyEx(trimmed_wavepad, cv.MORPH_ERODE, kernel, iterations = closing_iterations)
        closed_wavepad = cv.morphologyEx(closed_wavepad, cv.MORPH_CLOSE, kernel, iterations = closing_iterations)

    # Find correct sized objects
    logger.info(f"Finding Correct Sized Objects with min size: {min_obj_size}")
    correct_sized_wavepad, avg_obj_area = find_correct_sized_obj(closed_wavepad, min_obj_size)
    logger.info(f"Average Object Area: {avg_obj_area} for {direction} wavepad")

    # Find the center line
    center_line = find_center_line(correct_sized_wavepad, poly_degree, direction, logger)

    # Impute the skeleton
    imputed_skel = impute_skel(center_line, direction, logger)

    return imputed_skel
