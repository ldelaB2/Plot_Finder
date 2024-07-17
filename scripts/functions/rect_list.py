import numpy as np
import cv2 as cv
from tqdm import tqdm

from classes.rectangles import rectangle
from functions.image_processing import four_2_five_rect
from functions.distance_optimize import distance_optimize
from functions.display import dialate_skel, disp_quadratic_optimization
from functions.optimization import compute_model
from functions.display import disp_rectangles
from matplotlib import pyplot as plt
from functions.distance_optimize import compute_spiral_path, bfs_distance
import multiprocessing as mp
import copy

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
    # Pull the params
    ncores = opt_param_dict["ncore"]
    nstart = opt_param_dict["nstart"]
    opt_param_dict['model'] = model

    start_indx = np.random.choice(len(rect_list), nstart, replace = False)
    start_points = []
    for indx in start_indx:
        tmp = (rect_list[indx].range, rect_list[indx].row)
        start_points.append(tmp)

    print("Starting Sparse Optimization")

    with mp.Pool(processes = ncores) as pool:
        results = pool.map(bfs_distance, [(copy.copy(rect_list), opt_param_dict, start_point) for start_point in start_points])

    bfs_distance((rect_list, opt_param_dict, start_points[0]))
    print("")

    
    
    return

def final_optimize(rect_list, opt_param_dict, initial_model):
    # Pre Process the rectangles
    preprocess_rect_list(rect_list, opt_param_dict)
    
    # Define the parameters
    kernel_radi = opt_param_dict['kernel_radi']
    model = initial_model

   
    
    print(f"Starting Final Position Optimization")
    epoch = 1
    while epoch <= opt_param_dict['max_epoch']:
        print(f"Epoch {epoch}")
        xy_update = []
        hw_update = []
        theta_update = []

        # Optimize the rectangles
        for rect in tqdm(rect_list, desc = "Position Optimization"):
            test = rect.optimize_xy(model, opt_param_dict)
        
        print("T")
         # Find the outliers
        identify_outliers(rect_list, kernel_radi)
            #xy_update.append(rect.optimize_XY(model, opt_param_dict, display = True))
            #hw_update.append(rect.optimize_HW(model, opt_param_dict))
            #theta_update.append(rect.optimize_theta(model, opt_param_dict))


        print(f"Update Stats: \n"
            f"  Position: {np.sum(xy_update)} / {len(rect_list)}\n"
            f"  Height / Width: {np.sum(hw_update)} / {len(rect_list)}\n"
            f"  Theta: {np.sum(theta_update)} / {len(rect_list)}\n")
       
        # Distance Optimization
        distance_optimize(rect_list, kernel_radi, weight = .5, update = True)
        epoch += 1
        


    print("Finished Final Optimization")
        
    return




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

