from classes.rectangles import rectangle
from functions.rectangle import compute_score
import numpy as np
from functions.image_processing import create_unit_square
from functions.display import disp_spiral_path
from tqdm import tqdm
import cv2 as cv
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from functions.general import bindvec


def build_rect_list(rect_list, img):
        output_list = []
        # Find the mean width and height
        mean_width = np.mean([rect[2] for rect in rect_list])
        mean_height = np.mean([rect[3] for rect in rect_list])

        # Round the mean width and height
        mean_width = np.round(mean_width).astype(int)
        mean_height = np.round(mean_height).astype(int)

        # Create the unit square
        unit_sqr = create_unit_square(mean_width, mean_height)

        # Create the rectangles
        for rect in rect_list:
            rect[2] = mean_width
            rect[3] = mean_height

            rect = rectangle(rect)
            rect.img = img
            rect.unit_sqr = unit_sqr
            output_list.append(rect)

        return output_list

def compute_model(rect_list, initial = False, threshold = False):
    if len(rect_list[0].img.shape) == 2:
        model = np.zeros((rect_list[0].height, rect_list[0].width))
    else:
        model = np.zeros((rect_list[0].height, rect_list[0].width, rect_list[0].img.shape[2]))
    
    for rect in rect_list:
        sub_img = rect.create_sub_image()
        model += sub_img

    model = np.round(model / len(rect_list)).astype(np.uint8)
    
    return model

def sparse_optimize_list(rect_list, model, opt_param_dict, txt = "Optimizing Rectangles"):
    # Pull the parameters
    x_radi = opt_param_dict['x_radi']
    y_radi = opt_param_dict['y_radi']
    num_points = opt_param_dict['quadratic_num_points']

    num_sample = np.sqrt(num_points).astype(int)

    # Create the test points
    x = np.round(np.linspace(-x_radi, x_radi, num_sample)).astype(int)
    y = np.round(np.linspace(-y_radi, y_radi, num_sample)).astype(int)

    # Create the meshgrid
    X, Y = np.meshgrid(x, y)
    test_points = np.column_stack((X.ravel(), Y.ravel()))

    # Remove the origin if it exists
    test_points = test_points[np.where((test_points[:,0] != 0) | (test_points[:,1] != 0))]

    # Only keep unique values
    test_points = np.unique(test_points, axis = 0)

    # Push to the dictionary
    opt_param_dict['test_points'] = test_points

    # Pull the threshold
    threshhold = opt_param_dict['threshhold']
    max_epoch = opt_param_dict['max_epoch']

    # Define the kernel radi size
    kernel_radi = [2, 2]
    
    num_updated = np.inf
    epoch = 1
    model = bindvec(model)
    while num_updated > threshhold and epoch <= max_epoch:
        print(f"{txt} Starting Epoch {epoch}")

        results = []
        for rect in rect_list:
            if not rect.initial_opt:
                tmp_flag = rect.optomize_rectangle(model, opt_param_dict)
            else:
                tmp_flag = False

            if not tmp_flag:
                rect.initial_opt = True

            results.append(tmp_flag)

        num_updated = np.sum(results)
        print(f"Updated {num_updated} rectangles")

        distance_optimize(rect_list, kernel_radi, weight = .8)

        epoch += 1
    
    return

def final_optimize_list(rect_list, opt_param_dict):
    # Pull the parameters
    x_radi = opt_param_dict['x_radi']
    y_radi = opt_param_dict['y_radi']
    t_radi = opt_param_dict['theta_radi']
    num_points = opt_param_dict['quadratic_num_points']

    num_sample = np.power(num_points, 1/3).astype(int)

    # Create the test points
    x = np.round(np.linspace(-x_radi, x_radi, num_sample)).astype(int)
    y = np.round(np.linspace(-y_radi, y_radi, num_sample)).astype(int)
    t = np.round(np.linspace(-t_radi, t_radi, num_sample)).astype(int)

    # Create the meshgrid
    X, Y, T = np.meshgrid(x, y, t)
    test_points = np.column_stack((X.ravel(), Y.ravel(), T.ravel()))

    # Remove the origin if it exists
    test_points = test_points[np.where((test_points[:,0] != 0) | (test_points[:,1] != 0) | (test_points[:,2] != 0))]

    # Only keep unique values
    test_points = np.unique(test_points, axis = 0)

    # Push to the dictionary
    opt_param_dict['test_points'] = test_points

    # Pull the threshold
    threshhold = opt_param_dict['threshhold']
    max_epoch = opt_param_dict['max_epoch']
    kernel_radi = [2, 2]

    for epoch in range(max_epoch):
        print(f"Final Optimization Starting Epoch {epoch + 1}/{max_epoch}")
        model = compute_model(rect_list)
        total = len(rect_list)
        results = []

        for rect in tqdm(rect_list, total = total, desc = "Final Optimization"):
            tmp_flag = rect.optomize_rectangle(model, opt_param_dict)
            results.append(tmp_flag)
    
        num_updated = np.sum(results)
        print(f"Updated {num_updated} rectangles")

        distance_optimize(rect_list, kernel_radi, weight = .5)
        
    return


def set_range_row(rect_list):
    ranges = []
    rows = []
    center_x = []
    center_y = []
    for rect in rect_list:
        ranges.append(rect.range)
        rows.append(rect.row)
        center_x.append(rect.center_x)
        center_y.append(rect.center_y)
    
    unique_ranges = np.unique(ranges)
    unique_rows = np.unique(rows)
    avg_rng_y = []
    for rng in unique_ranges:
        indx = np.where(np.array(ranges) == rng)[0]
        avg_rng_y.append(np.mean(np.array(center_y)[indx]))

    avg_row_x = []
    for row in unique_rows:
        indx = np.where(np.array(rows) == row)[0]
        avg_row_x.append(np.mean(np.array(center_x)[indx]))

    sorted_ranges = unique_ranges[np.argsort(avg_rng_y)]
    sorted_rows = unique_rows[np.argsort(avg_row_x)]

    for rect in rect_list:
        rng_indx = np.where(sorted_ranges == rect.range)[0][0]
        row_indx = np.where(sorted_rows == rect.row)[0][0]
        rect.range = rng_indx
        rect.row = row_indx
        rect.ID = f"{rect.range}_{rect.row}"

    return

def geometric_median(points, weights=None, tol = 1e-2):
    points = np.asarray(points)
    if weights is None:
        weights = np.ones(len(points))
    else:
        weights = np.asarray(weights)
    
    guess = np.mean(points, axis = 0)

    while True:
        distances = np.linalg.norm(points - guess, axis=1)
        nonzero = (distances != 0)
        
        if not np.any(nonzero):
            return guess
        
        w = weights[nonzero] / distances[nonzero]
        new_guess = np.sum(points[nonzero] * w[:, None], axis=0) / np.sum(w)
        
        if np.linalg.norm(new_guess - guess) < tol:
            return new_guess
        
        guess = new_guess

def find_expected_center(current_rect, neighbor_rect):
    # pull the values
    cnt_rng = current_rect.range
    cnt_row = current_rect.row
    nbr_rng = neighbor_rect.range
    nbr_row = neighbor_rect.row
    nbr_center = np.array([neighbor_rect.center_x, neighbor_rect.center_y])

    # Find the distance between the rectangles
    rng_away = abs(cnt_rng - nbr_rng)
    row_away = abs(cnt_row - nbr_row)

    if nbr_row > cnt_row:
        row_away = -row_away
    if nbr_rng > cnt_rng:
        rng_away = -rng_away


    # Find the expected center
    dx = row_away * neighbor_rect.width
    dy = rng_away * neighbor_rect.height

    # Rotate the vector
    theta = np.radians(neighbor_rect.theta)
    dx_rot = dx * np.cos(theta) - dy * np.sin(theta)
    dy_rot = dx * np.sin(theta) + dy * np.cos(theta)

    # Round the values
    dx_rot = np.round(dx_rot).astype(int)
    dy_rot = np.round(dy_rot).astype(int)

    # Find the expected center
    expected_center = nbr_center + np.array([dx_rot, dy_rot])

    return expected_center

def compute_spiral_path(rect_list):
    def find_center(rect_list):
        # Get the range and row values
        ranges = np.array([rect.range for rect in rect_list])
        rows = np.array([rect.row for rect in rect_list])

        # Get the unique range and row values
        unique_ranges = np.unique(ranges)
        unique_rows = np.unique(rows)

        center_range = np.round(np.median(unique_ranges)).astype(int)
        center_row = np.round(np.median(unique_rows)).astype(int)

        indx = np.where((ranges == center_range) & (rows == center_row))[0][0]
        center_x = rect_list[indx].center_x
        center_y = rect_list[indx].center_y

        return [center_x, center_y]

    def compute_polar_coordinates(rect_list, center):
        # Compute the polar coordinates
        polar_coords = []
        for rect in rect_list:
            dx = rect.center_x - center[0]
            dy = rect.center_y - center[1]
            rng = rect.range
            row = rect.row
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            polar_coords.append([r, theta, rng, row])

        return polar_coords
    
    def sort_polar_coordinates(polar_coords):
        polar_coords = np.array(polar_coords)
        distance = polar_coords[:,0]
        angle = polar_coords[:,1]
        # Sort in descending order
        sorted_indx = np.lexsort((angle, distance))
        sorted_points = polar_coords[sorted_indx]
        rng_row = sorted_points[:,2:]
        rng_row = rng_row.astype(int)
        return rng_row

    center_coord = find_center(rect_list)
    polar_coords = compute_polar_coordinates(rect_list, center_coord)
    path = sort_polar_coordinates(polar_coords)

    return path


def compute_neighbors(rect_list, kernel):
    # Find min and max
    ranges = np.array([rect.range for rect in rect_list])
    rows = np.array([rect.row for rect in rect_list])
    max_range = np.max(ranges)
    min_range = np.min(ranges)
    max_row = np.max(rows)
    min_row = np.min(rows)
    for rect in rect_list:
        # Find self range and row
        rng = rect.range
        row = rect.row

        # Find the neighbors
        rng_neighbors = np.arange(rng - kernel[0], rng + kernel[0] + 1)
        row_neighbors = np.arange(row - kernel[1], row + kernel[1] + 1)

        # Clip the neighbors to valid values
        rng_neighbors = np.unique(np.clip(rng_neighbors, min_range, max_range))
        row_neighbors = np.unique(np.clip(row_neighbors, min_row, max_row))

        # Create the neighbor list
        x, y = np.meshgrid(rng_neighbors, row_neighbors)
        tmp_neighbors = np.column_stack((x.ravel(), y.ravel()))
        
        # Remove self from neighbors
        self_indx = np.where((tmp_neighbors[:,0] == rng) & (tmp_neighbors[:,1] == row))[0]
        tmp_neighbors = np.delete(tmp_neighbors, self_indx, axis = 0)
        tmp_neighbors = tmp_neighbors.tolist()
        rect.neighbors = tmp_neighbors

    return

def distance_optimize(rect_list, kernel, weight = .5):
    # Compute the spiral path
    spiral_path = compute_spiral_path(rect_list)

    #Compute the neighbors
    compute_neighbors(rect_list, kernel)

    # Find the range and row values
    ranges = np.array([rect.range for rect in rect_list])
    rows = np.array([rect.row for rect in rect_list])

    for point in spiral_path:
        path_rng = point[0]
        path_row = point[1]
        indx = np.where((ranges == path_rng) & (rows == path_row))[0][0]
        current_rect = rect_list[indx]
        neighbors = current_rect.neighbors
        expected_centers = []
        for neighbor in neighbors:
            # Find the neighbor
            nbr_rng = neighbor[0]
            nbr_row = neighbor[1]
            neighbor_indx = np.where((ranges == nbr_rng) & (rows == nbr_row))[0][0]
            neighbor_rect = rect_list[neighbor_indx]

            # Find the expected center
            expected_center = find_expected_center(current_rect, neighbor_rect)
            expected_centers.append(expected_center)
        
        # Compute the geometric median
        new_center = geometric_median(expected_centers)
        new_center = np.round(new_center).astype(int)

        dx = new_center[0] - current_rect.center_x
        dy = new_center[1] - current_rect.center_y

        dx = np.round(dx * weight).astype(int)
        dy = np.round(dy * weight).astype(int)

        current_rect.center_x += dx
        current_rect.center_y += dy

    print("Finished optimizing distance")
    return
    
