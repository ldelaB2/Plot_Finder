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

        for k in range(num_rows - 2):
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
            if not edge:
                tmp1_rect.range = min_val - 1
                tmp2_rect.range = max_val + 1
        elif direction == 'row':
            tmp1_rect.center_x = tmp1_rect.center_x - delta
            tmp2_rect.center_x = tmp2_rect.center_x + delta
            if not edge:
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

def remove_rectangles(total_list, remove_list):

    remove_range = [rect.range for rect in remove_list]
    remove_row = [rect.row for rect in remove_list]

    total_range = [rect.range for rect in total_list]
    total_row = [rect.row for rect in total_list]

    remove_indx = np.where(np.isin(total_range, remove_range) & np.isin(total_row, remove_row))[0]
    output = [total_list[indx] for indx in range(len(total_list)) if indx not in remove_indx]

    return output

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

def check_within_img(min_list, max_list, img_shape):
    min_center_x = np.mean([rect.center_x for rect in min_list])
    max_center_x = np.mean([rect.center_x for rect in max_list])
    min_center_y = np.mean([rect.center_y for rect in min_list])
    max_center_y = np.mean([rect.center_y for rect in max_list])

    min_score, max_score = 1, 1

    if min_center_x < 0:
        min_score = np.inf
    if max_center_x > img_shape[1]:
        max_score = np.inf
    if min_center_y < 0:
        min_score = np.inf
    if max_center_y > img_shape[0]:
        max_score = np.inf

    return min_score, max_score

def compare_next_to_current(rect_list, model, direction):
    # Find current edge
    current_min, current_max = find_next_rect(rect_list, direction, edge = True)
    # Find next set
    next_min, next_max = find_next_rect(rect_list, direction, edge = False)
    
    # Compute the current scores
    current_min_score = compute_score(current_min, model, method = "cosine")
    current_max_score = compute_score(current_max, model, method = "cosine")

    # Compute the next scores
    next_max_score = compute_score(next_max, model, method = "cosine")
    next_min_score = compute_score(next_min, model, method = "cosine")

    # Check if the mean center is within the image for the next row
    next_min_mult, next_max_mult = check_within_img(next_min, next_max, rect_list[0].img.shape)
    next_max_score = next_max_score * next_max_mult
    next_min_score = next_min_score * next_min_mult

    # Compare the next min to current max
    if next_min_score < current_max_score:
        drop_list = current_max
        add_list = next_min
        update_flag = True

    # Compare next max to current min
    elif next_max_score < current_min_score:
        drop_list = current_min
        add_list = next_max
        update_flag = True

    else:
        update_flag = False
        add_list = []
        drop_list = []

    return update_flag, add_list, drop_list

def compute_score(rect_list, model, method = "euclidean"):
    scores = []

    if method == "cosine":
        model_vec = model.flatten()
        model_norm = np.linalg.norm(model_vec)

        for rect in rect_list:
            subI = rect.create_sub_image()
            img_vec = subI.flatten()
            img_norm = np.linalg.norm(img_vec)

            if img_norm == 0:
                print("Zero norm image")
                scores.append(np.inf)
            else:
                cosine_similarity = np.dot(model_vec, img_vec) / (model_norm * img_norm)
                scores.append(cosine_similarity)

    elif method == "euclidean":
        for rect in rect_list:
            subI = rect.create_sub_image()
            scores.append(np.linalg.norm(subI - model))

    else:
        print("Invalid method for computing score")
        return
    
    final_score = np.median(scores)

    return final_score