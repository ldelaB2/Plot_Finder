import numpy as np

def distance_optimize(rect_list, kernel, weight = .5, update = False):
    # Compute the spiral path
    spiral_path = compute_spiral_path(rect_list)

    #Compute the neighbors
    compute_neighbors(rect_list, kernel)

    # Find the range and row values
    ranges = np.array([rect.range for rect in rect_list])
    rows = np.array([rect.row for rect in rect_list])
    output = []

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

        if update:
            current_rect.center_x += np.round(weight * dx).astype(int)
            current_rect.center_y += np.round(weight * dy).astype(int)
        else:
            tmp_output = [current_rect.range, current_rect.row, dx, dy]
            output.append(tmp_output)

    print("Finished optimizing distance")
    return output
    


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