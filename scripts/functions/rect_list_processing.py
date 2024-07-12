import numpy as np
import copy

from functions.optimization import compute_score_list
from functions.rect_list import sparse_optimize


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

        # Reset the flags
        tmp1_rect.added = True
        tmp2_rect.added = True


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

def remove_rectangles_from_list(total_list, remove_list):

    remove_range = [rect.range for rect in remove_list]
    remove_row = [rect.row for rect in remove_list]

    total_range = [rect.range for rect in total_list]
    total_row = [rect.row for rect in total_list]

    remove_indx = np.where(np.isin(total_range, remove_range) & np.isin(total_row, remove_row))[0]
    output = [total_list[indx] for indx in range(len(total_list)) if indx not in remove_indx]

    return output

def check_within_img(min_list, max_list):
    img_shape = min_list[0].img.shape
    min_center_x = np.mean([rect.center_x for rect in min_list])
    max_center_x = np.mean([rect.center_x for rect in max_list])
    min_center_y = np.mean([rect.center_y for rect in min_list])
    max_center_y = np.mean([rect.center_y for rect in max_list])

    min_flag, max_flag = True, True

    if min_center_x < 0 or min_center_y < 0:
        min_flag = False
    if max_center_x > img_shape[1] or max_center_y > img_shape[0]:
        max_flag = False

    return min_flag, max_flag

def compare_next_to_current(rect_list, model, direction, opt_param_dict):
    score_method = "L2"
    # Find current edge
    current_min, current_max = find_next_rect(rect_list, direction, edge = True)
    # Find next set
    next_min, next_max = find_next_rect(rect_list, direction, edge = False)

    # Check if the mean center is within the image for the next then optimize and compute the score
    next_min_flag, next_max_flag = check_within_img(next_min, next_max)
    if next_min_flag:
        sparse_optimize(next_min, model, opt_param_dict)
        next_min_score = compute_score_list(next_min, model, method = score_method)
    else:
        next_min_score = np.inf

    if next_max_flag:
        sparse_optimize(next_max, model, opt_param_dict)
        next_max_score = compute_score_list(next_max, model, method = score_method)
    else:
        next_max_score = np.inf

    # Compute the current scores
    current_min_score = compute_score_list(current_min, model, method = score_method)
    current_max_score = compute_score_list(current_max, model, method = score_method)

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

def remove_rectangles(rect_list, direction, num_2_remove, model):
    for _ in range(num_2_remove):
        # Find the current min, max rectangles
        min_list, max_list = find_next_rect(rect_list, direction, edge = True)

        # Computing the scores
        min_score = compute_score_list(min_list, model, method = "L1")
        max_score = compute_score_list(max_list, model, method = "L1")

        if min_score <= max_score:
            rect_list = remove_rectangles_from_list(rect_list, max_list)
            print(f"Removed Max {direction}")
        else:
            rect_list = remove_rectangles_from_list(rect_list, min_list)
            print(f"Removed Min {direction}")

    return rect_list

def add_rectangles(rect_list, direction, num_2_add, model, opt_param_dict):
    score_method = "L1"
    for cnt in range(num_2_add):
        # Find the next rectangles
        min_list, max_list = find_next_rect(rect_list, direction, edge = False)

        # Check to make sure the rectangles are within the image
        min_flag, max_flag = check_within_img(min_list, max_list)

        if min_flag:
            # Optimize the rectangles
            sparse_optimize(min_list, model, opt_param_dict)
            # Compute the score
            min_score = compute_score_list(min_list, model, method = score_method)
        else:
            min_score = np.inf
            print("Min Rectangles are out of bounds")

        if max_flag:
            # Optimize the rectangles
            sparse_optimize(max_list, model, opt_param_dict)
            # Compute the score
            max_score = compute_score_list(max_list, model, method = score_method)
        else:
            max_score = np.inf
            print("Max Rectangles are out of bounds")


        # Adding the rectangles with min score
        if min_score >= max_score:
            print(f"Adding Max {direction}")
            for rect in max_list:
                rect_list.append(rect)
        else:
            print(f"Adding Min {direction}")
            for rect in min_list:
                rect_list.append(rect)

    print(f"Finished adding {num_2_add} {direction}(s)")

    return rect_list

def double_check(rect_list, direction, model, opt_param_dict):
    # Checking to make sure we found the correct ranges and rows
    flag = True
    update_cnt = 0

    while flag:
        update_flag, next_best_list, current_best_list = compare_next_to_current(rect_list, model, direction, opt_param_dict)
        if update_flag:
            # Remove the current best list and add the next best list
            rect_list = remove_rectangles_from_list(rect_list, current_best_list)
            for rect in next_best_list:
                rect_list.append(rect)

            update_cnt += 1
        else:
            flag = False
    
    print(f"Shifted {update_cnt} {direction}(s)")

    return rect_list
