import numpy as np
import cv2 as cv
from functions.display import dialate_skel
from functions.image_processing import create_affine_frame
import copy
from sklearn.cluster import KMeans

def build_rectangles(range_skel, col_skel):
    num_ranges, ld_range, _, _ = cv.connectedComponentsWithStats(dialate_skel(range_skel))
    num_rows, ld_row, _, _ = cv.connectedComponentsWithStats(dialate_skel(col_skel))

    cp = np.argwhere(range_skel.astype(bool) & col_skel.astype(bool))

    rect = []
    for e in range(1, num_ranges - 1):
        # Find the top and bottom points
        top_points = cp[np.where(ld_range[cp[:,0], cp[:,1]] == e)[0]]
        bottom_points = cp[np.where(ld_range[cp[:,0], cp[:,1]] == e + 1)[0]]

        # Sort the points based on the x values
        top_points = top_points[np.argsort(top_points[:,1])]
        bottom_points = bottom_points[np.argsort(bottom_points[:,1])]

        for k in range(top_points.shape[0] - 1):
            top_left = top_points[k,:]
            top_right = top_points[k + 1,:]
            bottom_left = bottom_points[k,:]
            bottom_right = bottom_points[k + 1,:]
            points = [top_left, top_right, bottom_left, bottom_right]
            tmp_list = four_2_five_rect(points)
            tmp_list = np.append(tmp_list, [e, k])
            rect.append(tmp_list)
    
    num_ranges = num_ranges - 2
    num_rows = num_rows - 2

    return rect, num_ranges, num_rows

def four_2_five_rect(points):
    top_left, top_right, bottom_left, bottom_right = points
    w1 = np.linalg.norm(top_left - top_right)
    w2 = np.linalg.norm(bottom_left - bottom_right)
    h1 = np.linalg.norm(top_left - bottom_left)
    h2 = np.linalg.norm(top_right - bottom_right)
    width = ((w1 + w2)/2).astype(int)
    height = ((h1 + h2)/2).astype(int)

    # Computing theta
    dx1 = top_right[1] - top_left[1]
    dy1 = top_right[0] - top_left[0]
    dx2 = bottom_right[1] - bottom_left[1]
    dy2 = bottom_right[0] - bottom_left[0]

    theta1 = np.arctan2(dy1, dx1)
    theta2 = np.arctan2(dy2, dx2)
    theta = (theta1 + theta2) / 2
    theta = np.degrees(theta)
    theta = np.round(theta).astype(int)

    center = np.mean((top_left,top_right,bottom_left,bottom_right), axis = 0).astype(int)
    rect = np.append(center, [width, height, theta])

    return rect

def five_2_four_rect(points):
    center_x, center_y, width, height, theta = points
    points = np.array([[-1,1,1], [1,1,1], [1,-1,1], [-1,-1,1]])
    aff_mat = create_affine_frame(center_x, center_y, np.radians(theta), width, height)
    corner_points = np.dot(aff_mat, points.T).T
    corner_points = corner_points[:,:2].astype(int)

    return corner_points

def find_next_rect(rect_list, direction, edge = False):
    if direction == 'row':
        values = np.array([rect.row for rect in rect_list])
        delta = rect_list[0].width
    elif direction == 'range':
        values = np.array([rect.range for rect in rect_list])
        delta = rect_list[0].height
    
    if edge:
        delta = 0

    unique_values = np.unique(values)
    min_val = np.min(unique_values)
    max_val = np.max(unique_values)
    min_val_rect = np.where(values == min_val)[0]
    max_val_rect = np.where(values == max_val)[0]
    min_list = []
    max_list = []

    for e in range(len(min_val_rect)):
        tmp1_rect = copy.copy(rect_list[min_val_rect[e]])
        tmp2_rect = copy.copy(rect_list[max_val_rect[e]])

        if direction == 'range':
            tmp1_rect.center_y = tmp1_rect.center_y - delta
            tmp2_rect.center_y = tmp2_rect.center_y + delta
            tmp1_rect.range = min_val - 1
            tmp2_rect.range = max_val + 1
        elif direction == 'row':
            tmp1_rect.center_x = tmp1_rect.center_x - delta
            tmp2_rect.center_x = tmp2_rect.center_x + delta
            tmp1_rect.row = min_val - 1
            tmp2_rect.row = max_val + 1

        min_list.append(tmp1_rect)
        max_list.append(tmp2_rect)

    return min_list, max_list

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
        new_rows[indx] = row_ordered_indices[i]

    # Make sure range clusters are in order
    range_centroids = range_cluster.cluster_centers_.flatten()
    range_ordered_indices = np.argsort(range_centroids)
    new_ranges = np.zeros_like(ranges)
    for i in range(num_ranges):
        indxs = np.where(ranges == i)[0]
        new_ranges[indxs] = range_ordered_indices[i]

    # Set the range and row values for each rectangle
    for e, rect in enumerate(rect_list):
        rect.range = new_ranges[e]
        rect.row = new_rows[e]

    return rect_list

def remove_rectangles(total_list, remove_list):
    remove_range = [rect.range for rect in remove_list]
    remove_row = [rect.row for rect in remove_list]

    total_range = [rect.range for rect in total_list]
    total_row = [rect.row for rect in total_list]

    remove_indx = np.where(np.isin(total_range, remove_range) & np.isin(total_row, remove_row))[0]
    output = [total_list[indx] for indx in range(len(total_list)) if indx not in remove_indx]

    return output