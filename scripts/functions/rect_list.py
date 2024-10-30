import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans

from classes.rectangles import rectangle
from functions.image_processing import four_2_five_rect
from functions.display import dialate_skel

def build_rect_list(range_skel, row_skel, img):
        rect_list, num_ranges, num_rows = build_rectangles(range_skel, row_skel)

        output_list = []
        # Find the mean width and height
        mean_width = np.mean([rect[2] for rect in rect_list])
        mean_height = np.mean([rect[3] for rect in rect_list])

        # Round the mean width and height
        mean_width = np.round(mean_width).astype(int)
        mean_height = np.round(mean_height).astype(int)

        # Create the rectangles
        for rect in rect_list:
            rect[2] = mean_width
            rect[3] = mean_height

            rect = rectangle(rect)
            rect.img = img
            output_list.append(rect)

        return output_list, num_ranges, num_rows, mean_width, mean_height

def build_rectangles(range_skel, col_skel):
    dialated_range = dialate_skel(range_skel, 3)
    dialated_col = dialate_skel(col_skel, 3)

    num_ranges, ld_range, _, _ = cv.connectedComponentsWithStats(dialated_range)
    num_rows, ld_row, _, _ = cv.connectedComponentsWithStats(dialated_col)
    corner_points = np.where((ld_range != 0) & (ld_row != 0))

    def find_point(e, k):
        indx = np.where((ld_range[corner_points] == e) & (ld_row[corner_points] == k))
        points = np.array([corner_points[0][indx], corner_points[1][indx]])
        points = points.T
        center = np.round(np.mean(points, axis = 0)).astype(int)
        return center
        
    rect = []
    for e in range(1, num_ranges - 1):
        for k in range(1, num_rows - 1):
            top_left = find_point(e, k)
            top_right = find_point(e, k + 1)
            bottom_left = find_point(e + 1, k)
            bottom_right = find_point(e + 1, k + 1)

            points = [top_left, top_right, bottom_left, bottom_right]
            tmp_list = four_2_five_rect(points)
            tmp_list = np.append(tmp_list, [e, k])
            rect.append(tmp_list)

    
    num_ranges = num_ranges - 2
    num_rows = num_rows - 2

    return rect, num_ranges, num_rows

def set_range_row(rect_list):
    range_list = np.array([rect.range for rect in rect_list])
    row_list = np.array([rect.row for rect in rect_list])
    unique_ranges = np.unique(range_list)
    unique_rows = np.unique(row_list)

    rng_cnt = 1
    for range_val in unique_ranges:
        row_cnt = 1
        for row_val in unique_rows:
            indx = np.where((range_list == range_val) & (row_list == row_val))[0]
            if len(indx) > 1:
                print(f"Range {rng_cnt} Row {row_cnt} has {len(indx)} rectangles")
                return
            else:
                indx = indx[0]
                rect_list[indx].range = rng_cnt
                rect_list[indx].row = row_cnt
            row_cnt += 1
        rng_cnt += 1

    return

def set_id(rect_list, start, flow):
    # Get all the ranges and rows
    all_ranges = [rect.range for rect in rect_list]
    all_rows = [rect.row for rect in rect_list]

    # Get the unique ranges and rows
    ranges = np.unique(all_ranges)
    rows = np.unique(all_rows)

    if start == "TR":
        rows = np.flip(rows)
    elif start == "BL":
        ranges = np.flip(ranges)
    elif start == "BR":
        rows = np.flip(rows)
        ranges = np.flip(ranges)
    elif start == "TL":
        pass
    else:
        print("Invalid Start Point for ID labeling")
        return rect_list
    
    # Flip the rows for snake pattern
    rows_flipped = np.flip(rows)
    range_flip = np.zeros_like(ranges).astype(bool)
    if flow == "linear":
        pass
    elif flow == "snake":
        # Flip odd ranges to create a snake pattern
        range_flip[1::2] = True
    else:
        print("Invalid Flow for ID labeling")
        return rect_list
    
    cnt = 1
    for idx, r in enumerate(ranges):
        if range_flip[idx]:
            tmp_rows = rows_flipped
        else:
            tmp_rows = rows

        for e in tmp_rows:
            indx = np.where((all_ranges == r) & (all_rows == e))[0][0]
            rect_list[indx].ID = cnt
            cnt += 1

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

def build_rect_list_points(points, num_ranges, num_rows, img, logger):
    points = np.array(points)

    # Cluster the range values
    range_data = points[:,0].reshape(-1,1)
    range_k_means = KMeans(n_clusters = num_ranges, n_init = 500).fit(range_data)
    range_centers = range_k_means.cluster_centers_
    range_labels = range_k_means.labels_

    # Assign the range labels
    sorted_ranges = np.argsort(range_centers, axis = 0)
    sorted_range_labels = np.copy(range_labels)

    for indx in range(num_ranges):
        sorted_range_labels[range_labels == sorted_ranges[indx]] = indx

    points = np.hstack((points, sorted_range_labels.reshape(-1,1)))

    # Assign the row labels
    labeled_ranges = []
    for indx in range(num_ranges):
        range_points = points[points[:,-1] == indx,:]
        
        if range_points.shape[0] != num_rows:
            logger.error(f"For range {indx} detected {range_points.shape[0]} rows, expected {num_rows}")
            logger.info("Make sure to check final labels by hand")
        
        sorted_range = range_points[np.argsort(range_points[:,1])]
        rows = np.arange(0, range_points.shape[0], 1)
        sorted_range = np.hstack((sorted_range, rows.reshape(-1,1)))
        labeled_ranges.append(sorted_range)

    labeled_ranges = np.vstack(labeled_ranges)

    # Create the rectangles
    output_list = []
    for rect in labeled_ranges:
        rect = rectangle(rect)
        rect.img = img
        output_list.append(rect)

    return output_list









