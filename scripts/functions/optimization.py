from classes.rectangles import rectangle
from functions.rectangle import compute_score
import numpy as np
from functions.image_processing import create_unit_square
from functions.display import disp_spiral_path
from tqdm import tqdm
import cv2 as cv

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

def compute_model(rect_list, initial = False):
    # Create the model object
    if len(rect_list[0].img.shape) == 2:
        model = np.zeros((rect_list[0].height, rect_list[0].width))
    else:
        model = np.zeros((rect_list[0].height, rect_list[0].width, rect_list[0].img.shape[2]))

    if initial:
        # Preallocate memory
        distance = np.zeros((len(rect_list), len(rect_list)))

        # Compute the distance matrix (only lower triangle because symetric)
        for cnt1, rect1 in enumerate(rect_list):
            img1 = rect1.create_sub_image()
            for cnt2 in range(cnt1, len(rect_list)):
                rect2 = rect_list[cnt2]
                img2 = rect2.create_sub_image()
                tmp_score = compute_score(img1, img2, method = "cosine")
                distance[cnt1, cnt2] = tmp_score
        
        # Filling in the other side of symetric matrix
        distance = distance + distance.T

        # Compute the sum and median
        distance_sum = np.sum(distance, axis = 1)
        distance_median = np.median(distance_sum)
        
        # Find rects below the median
        good_rect_indx = np.argwhere(distance_sum <= distance_median)

        # Compute the model
        for indx in good_rect_indx:
            rect = rect_list[indx[0]]
            subI = rect.create_sub_image()
            model += subI

        model = np.round((model / len(good_rect_indx))).astype(np.uint8)

    else:
        for rect in rect_list:
            subI = rect.create_sub_image()
            model += subI
        
        model = np.round((model / len(rect_list))).astype(np.uint8)
    
    # Filter the model
    _, binary_model = cv.threshold(model, 0, 1, cv.THRESH_OTSU)
    model = model * binary_model

    print("Finished computing model")
    return model

def sparse_optimize_list(rect_list, model, opt_param_dict, txt = "Optimizing Rectangles"):
    # Pull the parameters
    x_radi = opt_param_dict['x_radi']
    y_radi = opt_param_dict['y_radi']
    num_points = opt_param_dict['quadratic_num_points']

    # Create the test points
    x = np.round(np.linspace(-x_radi, x_radi, num_points)).astype(int)
    y = np.round(np.linspace(-y_radi, y_radi, num_points)).astype(int)

    # Remove the zeros
    x = x[x != 0]
    y = y[y != 0]

    # Only keep unique values
    x = np.unique(x)
    y = np.unique(y)

    X, Y = np.meshgrid(x, y)
    test_points = np.column_stack((X.ravel(), Y.ravel()))
    test_points = np.hstack((test_points, np.zeros((X.size, 1))))
    # Push to the dictionary
    opt_param_dict['test_points'] = test_points
    opt_param_dict['test_y'] = y
    opt_param_dict['test_x'] = x

    # Pull the threshold
    threshhold = opt_param_dict['threshhold']
    max_epoch = opt_param_dict['max_epoch']
    
    num_updated = np.inf
    epoch = 1
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

        epoch += 1
    
    return


def set_range_row(rect_list, num_ranges, num_rows):
    center_x = np.array([rect.center_x for rect in rect_list])
    center_y = np.array([rect.center_y for rect in rect_list])

    row_cluster = KMeans(n_clusters = num_rows, n_init = 1000, max_iter = 200)
    rows = row_cluster.fit_predict(center_x.reshape(-1,1))

    range_cluster = KMeans(n_clusters = num_ranges, n_init = 1000, max_iter = 200)
    ranges = range_cluster.fit_predict(center_y.reshape(-1,1))

    # Make sure row clusters are in order
    row_centroids = row_cluster.cluster_centers_.flatten()
    row_ordered_indices = np.argsort(row_centroids)
    new_rows = np.zeros_like(rows)
    for i in range(num_rows):
        indx = np.where(rows == i)[0]
        new_indx = np.argwhere(row_ordered_indices == i)[0][0]
        new_rows[indx] = new_indx

    # Make sure range clusters are in order
    range_centroids = range_cluster.cluster_centers_.flatten()
    range_ordered_indices = np.argsort(range_centroids)
    new_ranges = np.zeros_like(ranges)
    for i in range(num_ranges):
        indx = np.where(ranges == i)[0]
        new_indx = np.argwhere(range_ordered_indices == i)[0][0]
        new_ranges[indx] = new_indx

    # Set the range and row values for each rectangle
    for e, rect in enumerate(rect_list):
        rect.range = new_ranges[e]
        rect.row = new_rows[e]
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
    rng_away = cnt_rng - nbr_rng
    row_away = cnt_row - nbr_row

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
    print("Finished computing spiral path")

    return path


def compute_neighbors(rect_list, kernel):
    # Find min and max
    ranges = np.array([rect.range for rect in rect_list])
    rows = np.array([rect.row for rect in rect_list])
    max_range = np.max(ranges)
    max_row = np.max(rows)
    for rect in rect_list:
        # Find self range and row
        rng = rect.range
        row = rect.row

        # Find the neighbors
        rng_neighbors = np.arange(rng - kernel[0], rng + kernel[0] + 1)
        row_neighbors = np.arange(row - kernel[1], row + kernel[1] + 1)

        # Clip the neighbors to valid values
        rng_neighbors = np.unique(np.clip(rng_neighbors, 1, max_range))
        row_neighbors = np.unique(np.clip(row_neighbors, 1, max_row))

        # Create the neighbor list
        x, y = np.meshgrid(rng_neighbors, row_neighbors)
        tmp_neighbors = np.column_stack((x.ravel(), y.ravel()))
        
        # Remove self from neighbors
        self_indx = np.where((tmp_neighbors[:,0] == rng) & (tmp_neighbors[:,1] == row))[0]
        tmp_neighbors = np.delete(tmp_neighbors, self_indx, axis = 0)
        tmp_neighbors = tmp_neighbors.tolist()
        rect.neighbors = tmp_neighbors

    print("Finished computing neighbors")
    return

def distance_optimize(rect_list, spiral_path):
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
        current_rect.center_x = new_center[0]
        current_rect.center_y = new_center[1]

    print("Finished optimizing distance")
    return
    

