import numpy as np
import copy

from functions.optimization import compute_template_score
from functions.display import disp_rectangles
from matplotlib import pyplot as plt
from functions.general import geometric_median
from functions.rect_list import compute_spiral_path

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
        tmp1_rect.clear()
        tmp2_rect.clear()
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

def compare_next_to_current(rect_list, direction, template_image, x_radi, y_radi, logger):
    # Find current edge
    current_min, current_max = find_next_rect(rect_list, direction, edge = True)
    # Find next set
    next_min, next_max = find_next_rect(rect_list, direction, edge = False)

    # Check if the mean center is within the image for the next then optimize and compute the score
    next_min_flag, next_max_flag = check_within_img(next_min, next_max)

    if next_min_flag:
        next_min_score = compute_template_score(next_min, template_image, x_radi, y_radi)
    else:
        next_min_score = np.inf
        logger.info(f"Next Min {direction} is out of bounds")

    if next_max_flag:
        next_max_score = compute_template_score(next_max, template_image, x_radi, y_radi)
    else:
        next_max_score = np.inf
        logger.info(f"Next Max {direction} is out of bounds")

    # Compute the current scores
    current_min_score = np.mean([rect.score for rect in current_min])
    current_max_score = np.mean([rect.score for rect in current_max])

    if current_min_score < current_max_score:
        # Compare the next min to current max
        if next_min_score < current_max_score:
            drop_list = current_max
            add_list = next_min
            update_flag = True
        else:
            update_flag = False

    elif current_max_score < current_min_score:
        # Compare next max to current min
        if next_max_score < current_min_score:
            drop_list = current_min
            add_list = next_max
            update_flag = True
        else:
            update_flag = False
          
    else:
        update_flag = False

    if not update_flag:
        add_list = []
        drop_list = []

    return update_flag, add_list, drop_list

def remove_rectangles(rect_list, direction, num_2_remove, logger):
    for _ in range(num_2_remove):
        # Find the current min, max rectangles
        min_list, max_list = find_next_rect(rect_list, direction, edge = True)

        # Computing the scores
        min_score = np.mean([rect.score for rect in min_list])
        max_score = np.mean([rect.score for rect in max_list])

        if min_score <= max_score:
            logger.info(f"Removing Max {direction}")
            rect_list = remove_rectangles_from_list(rect_list, max_list)
        else:
            logger.info(f"Removing Min {direction}")
            rect_list = remove_rectangles_from_list(rect_list, min_list)

    return rect_list

def add_rectangles(rect_list, direction, num_2_add, template_img, x_radi, y_radi, logger):
    for cnt in range(num_2_add):
        # Find the next rectangles
        min_list, max_list = find_next_rect(rect_list, direction, edge = False)

        # Check to make sure the rectangles are within the image
        min_flag, max_flag = check_within_img(min_list, max_list)

        if min_flag:
            # Optimize the rectangles
            min_score = compute_template_score(min_list, template_img, x_radi, y_radi)
                
        else:
            min_score = np.inf
            logger.info("Min Rectangles are out of bounds")

        if max_flag:
            max_score = compute_template_score(max_list, template_img, x_radi, y_radi)
        else:
            max_score = np.inf
            logger.info("Max Rectangles are out of bounds")


        # Adding the rectangles with min score
        if min_score > max_score:
            logger.info(f"Adding Max {direction}")
            for rect in max_list:
                rect_list.append(rect)
        else:
            logger.info(f"Adding Min {direction}")
            for rect in min_list:
                rect_list.append(rect)

    logger.info(f"Finished Adding {num_2_add} {direction}(s)")

    return rect_list

def double_check(rect_list, direction, template_image, x_radi, y_radi, logger):
    # Checking to make sure we found the correct ranges and rows
    flag = True
    update_cnt = 0

    while flag:
        update_flag, next_best_list, current_best_list = compare_next_to_current(rect_list, direction, template_image, x_radi, y_radi, logger)
        if update_flag:
            # Remove the current best list and add the next best list
            rect_list = remove_rectangles_from_list(rect_list, current_best_list)
            for rect in next_best_list:
                rect_list.append(rect)

            update_cnt += 1
        else:
            flag = False
    
    logger.info(f"Shifted {update_cnt} {direction}(s)")

    return rect_list

def compute_neighbors(rect_list, neighbor_radi):
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
        rng_neighbors = np.arange(rng - neighbor_radi, rng + neighbor_radi + 1)
        row_neighbors = np.arange(row - neighbor_radi, row + neighbor_radi + 1)

        # Clip the neighbors to valid values
        rng_neighbors = np.unique(np.clip(rng_neighbors, min_range, max_range))
        row_neighbors = np.unique(np.clip(row_neighbors, min_row, max_row))

        # Create the neighbor list
        x, y = np.meshgrid(rng_neighbors, row_neighbors)
        tmp_neighbors = np.column_stack((x.ravel(), y.ravel()))
        
        # Remove self from neighbors
        self_indx = np.where((tmp_neighbors[:,0] == rng) & (tmp_neighbors[:,1] == row))[0]
        tmp_neighbors = np.delete(tmp_neighbors, self_indx, axis = 0)
        tmp_neighbors = [tuple(nbr) for nbr in tmp_neighbors]
        rect.neighbors = tmp_neighbors

    return

def distance_optimize(rect_list, neighbor_radi, kappa, logger):
    logger.info("Starting Distance Optimization")

    # Compute the average neighbor distance
    avg_height = []
    avg_width = []
    n = len(rect_list)
    for i in range(n):
        for j in range(i + 1, n):
            rect1 = rect_list[i]
            rect2 = rect_list[j]

            rng_away = abs(rect1.range - rect2.range)
            row_away = abs(rect1.row - rect2.row)

            if rng_away != 0:
                y_away = abs(rect1.center_y - rect2.center_y)
                tmp_height = y_away / rng_away
                avg_height.append(tmp_height)
            
            if row_away != 0:
                x_away = abs(rect1.center_x - rect2.center_x)
                tmp_width = x_away / row_away
                avg_width.append(tmp_width)

    avg_height = np.round(np.mean(avg_height)).astype(int)
    avg_width = np.round(np.mean(avg_width)).astype(int)

    neighbor_distance = [avg_height, avg_width]

    #Compute the spiral path
    spiral_path = compute_spiral_path(rect_list)
    logger.info(f"Start Point: {spiral_path[0]}")
    # Compute the neighbors
    compute_neighbors(rect_list, neighbor_radi)


    # Set up for the first pass
    first_pass = [copy.copy(rect) for rect in rect_list]
    rect_dict = {(rect.range, rect.row): rect for rect in first_pass}
    distance_from_geometric_mean = []

    # Compute the geometric median for myself
    for point in spiral_path:
        # Myself
        me = (point[0], point[1])
        rect = rect_dict[me]
        # My neighbors
        my_neighbors = rect.neighbors

        # Save the output
        expected_centers = []
        for neighbor in my_neighbors:
            # Neighbor
            neighbor_rect = rect_dict[neighbor]
            # Where my neighbor thinks I should be
            expected_centers.append(neighbor_rect.compute_neighbor_position(neighbor_distance, me))
        
        expected_center = np.array([sublist[0] for sublist in expected_centers])
        # Compute the geometric median
        geometric_mean = np.round(geometric_median(expected_center)).astype(int)

        # Compute the distance from the geometric mean
        my_distance = np.linalg.norm(np.array([rect.center_x, rect.center_y]) - geometric_mean)

        # Update the rect center
        rect.center_x = geometric_mean[0]
        rect.center_y = geometric_mean[1]

        distance_from_geometric_mean.append(my_distance)

    # Create the sigmoid for scaling the weight of the distance in placement
    x0 = np.mean(distance_from_geometric_mean)
    
    logger.info(f"Mean Distance | K from Geometric Mean Sigmoid: {np.round(x0)} | {kappa}")
    
    # Compute the sigmoid values
    def sigmoid(x):
        y = 1/ (1 + np.exp(-kappa * (x - x0)))
        return y
    
    # Setup for the second pass
    second_pass = [copy.copy(rect) for rect in rect_list]
    rect_dict = {(rect.range, rect.row): rect for rect in second_pass}

    for point in spiral_path:
        # Myself
        me = (point[0], point[1])
        rect = rect_dict[me]
        # My neighbors
        my_neighbors = rect.neighbors

        # Save the output
        expected_centers = []
        for neighbor in my_neighbors:
            # Neighbor
            neighbor_rect = rect_dict[neighbor]
            # Where my neighbor thinks I should be
            expected_centers.append(neighbor_rect.compute_neighbor_position(neighbor_distance, me))
        
        expected_center = np.array([sublist[0] for sublist in expected_centers])
        expected_attributes = np.array([sublist[1] for sublist in expected_centers])

        # Compute the geometric median
        geometric_mean = np.round(geometric_median(expected_center)).astype(int)

        # Compute the distance from the geometric mean
        delta_center = geometric_mean - np.array([rect.center_x, rect.center_y])

        # Compute the weight
        my_distance = np.linalg.norm(delta_center)
        weight = sigmoid(my_distance)

        mean_attributes = np.mean(expected_attributes, axis = 0)
        my_attributes = np.array([rect.theta, rect.width, rect.height])
        delta_attributes = mean_attributes - my_attributes

        weighted_attributes = delta_attributes * weight
        weighted_delta = np.round(delta_center * weight).astype(int)

        # Update the rect center
        rect.center_x += weighted_delta[0]
        rect.center_y += weighted_delta[1]
        # Update the rect attributes
        rect.theta += np.round(weighted_attributes[0], 3)
        rect.width += np.round(weighted_attributes[1]).astype(int)
        rect.height += np.round(weighted_attributes[2]).astype(int)
        # Recompute the corner points
        rect.recompute_unit_sqr = True


    logger.info("Finished Distance Optimization")

    return second_pass







