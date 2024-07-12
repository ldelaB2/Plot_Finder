import numpy as np
import cv2 as cv
from tqdm import tqdm

from classes.rectangles import rectangle
from functions.image_processing import create_unit_square, four_2_five_rect
from functions.distance_optimize import distance_optimize
from functions.display import dialate_skel, disp_quadratic_optimization
from functions.optimization import compute_model, shrink_rect

def build_rect_list(range_skel, row_skel, img):
        rect_list, num_ranges, num_rows = build_rectangles(range_skel, row_skel)


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

        return output_list, num_ranges, num_rows

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

def sparse_optimize(rect_list, model, opt_param_dict):
    # Pull the parameters
    x_radi = opt_param_dict['x_radi']
    y_radi = opt_param_dict['y_radi']
    num_points = opt_param_dict['quadratic_num_points']

    # Compute the test points
    test_points, total_points = compute_points(x_radi, y_radi, num_points)

    # Push to the dictionary
    opt_param_dict['test_points'] = test_points

    # Pull the number of epochs and kernel radii
    max_epoch = opt_param_dict['max_epoch']
    kernel_radi = opt_param_dict['kernel_radi']

    print("Starting Sparse Optimization")
    results = []
    for rect in rect_list:
        tmp_flag = rect.optomize_rectangle(model, opt_param_dict)
        results.append(tmp_flag)

    num_updated = np.sum(results)
    print(f"Updated {num_updated} rectangles")

    distance_optimize(rect_list, kernel_radi, weight = .8, update = True)

    
    return

def final_optimize(rect_list, opt_param_dict):
    # Pull the parameters
    x_radi = opt_param_dict['x_radi']
    y_radi = opt_param_dict['y_radi']
    t_radi = opt_param_dict['theta_radi']
    num_points = opt_param_dict['quadratic_num_points']

    # Compute the test points
    test_points, total_points = compute_points(x_radi, y_radi, num_points)

    print(f"Using {test_points.shape[0]} test points")
    print(f"Using {total_points.shape[0]} total points")
    # Push to the dictionary
    opt_param_dict['test_points'] = test_points
    opt_param_dict['total_points'] = total_points
    kernel_radi = opt_param_dict['kernel_radi']
    
    print(f"Starting Final Optimization")
    model = compute_model(rect_list)
    total = len(rect_list)
    position_update = []
    theta_update = []

    for rect in tqdm(rect_list, total = total, desc = "Final Optimization"):
        pos_flag = rect.optomize_rectangle(model, opt_param_dict)
        theta_flag = rect.optomize_rectangle_theta(model, opt_param_dict)
        shrink_rect(rect, model, opt_param_dict)
        position_update.append(pos_flag)
        theta_update.append(theta_flag)

    print(f"Updated position of {np.sum(position_update)} rectangles")
    print(f"Updated theta of {np.sum(theta_update)} rectangles")
    #Recompute the model
    distance_optimize(rect_list, kernel_radi, weight = .5, update = True)

    print("Finished Final Optimization")
        
    return



def compute_points(x_radi, y_radi, num_points):
    y_ratio = y_radi / x_radi

    x_num_sample = np.power(num_points / y_ratio, 1/2)
    y_num_sample = (x_num_sample * y_ratio).astype(int)
    x_num_sample = x_num_sample.astype(int)

    # Create the test points
    x = np.round(np.linspace(-x_radi, x_radi, x_num_sample)).astype(int)
    y = np.round(np.linspace(-y_radi, y_radi, y_num_sample)).astype(int)

    # Create the meshgrid
    X, Y = np.meshgrid(x, y)
    test_points = np.column_stack((X.ravel(), Y.ravel()))

    # Remove the origin if it exists
    test_points = test_points[np.where((test_points[:,0] != 0) | (test_points[:,1] != 0))]

    # Only keep unique values
    test_points = np.unique(test_points, axis = 0)

    # Create the total points
    x_total = np.arange(-x_radi, x_radi, 2)
    y_total = np.arange(-y_radi, y_radi, 2)
    X, Y, = np.meshgrid(x_total, y_total)
    total_points = np.column_stack((X.ravel(), Y.ravel()))
    
    return test_points , total_points

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

