import numpy as np
import cv2 as cv
from functions.display import dialate_skel
from functions.image_processing import create_affine_frame
from copy import deepcopy

def build_rectangles(range_skel, col_skel):
    num_ranges, ld_range, _, _ = cv.connectedComponentsWithStats(dialate_skel(range_skel))
    
    range_bool = range_skel.astype(bool)
    col_bool = col_skel.astype(bool)
    cp = set(map(tuple, np.argwhere(col_bool & range_bool)))
    rect_list = []
    range_cnt = 0
    rect_cnt = 0

    for e in range(1, num_ranges - 1):
        top_indx = set(map(tuple, np.argwhere(ld_range == e)))
        bottom_indx = set(map(tuple, np.argwhere(ld_range == e + 1)))
        top_points = np.array(list(cp.intersection(top_indx)))
        bottom_points = np.array(list(cp.intersection(bottom_indx)))
        top_points = top_points[np.argsort(top_points[:,1])]
        bottom_points = bottom_points[np.argsort(bottom_points[:,1])]
        row_cnt = 0
        for k in range(top_points.shape[0] - 1):
            top_left = top_points[k,:]
            top_right = top_points[k + 1,:]
            bottom_left = bottom_points[k,:]
            bottom_right = bottom_points[k + 1,:]
            points = [top_left, top_right, bottom_left, bottom_right]
            rect = four_2_five_rect(points)
            position = [range_cnt, row_cnt]
            rect = np.append(rect, position)
            rect_list.append(rect)
            row_cnt += 1
            rect_cnt += 1

        range_cnt += 1

    return rect_list, range_cnt, row_cnt

def four_2_five_rect(points):
    top_left, top_right, bottom_left, bottom_right = points
    w1 = np.linalg.norm(top_left - top_right)
    w2 = np.linalg.norm(bottom_left - bottom_right)
    h1 = np.linalg.norm(top_left - bottom_left)
    h2 = np.linalg.norm(top_right - bottom_right)
    width = ((w1 + w2)/2).astype(int)
    height = ((h1 + h2)/2).astype(int)
    center = np.mean((top_left,top_right,bottom_left,bottom_right), axis = 0).astype(int)
    rect = np.append(center, [width, height, 0])

    return rect

def five_2_four_rect(points):
    center_x, center_y, width, height, theta = points
    points = np.array([[-1,1,1], [1,1,1], [1,-1,1], [-1,-1,1]])
    aff_mat = create_affine_frame(center_x, center_y, np.radians(theta), width, height)
    corner_points = np.dot(aff_mat, points.T).T
    corner_points = corner_points[:,:2].astype(int)

    return corner_points

def find_next_rect(rect_list_obj, direction):
    rect_list = rect_list_obj.rect_list
    if direction == 'row':
        values = np.array([rect.row for rect in rect_list])
        delta = rect_list_obj.width
    elif direction == 'range':
        values = np.array([rect.range for rect in rect_list])
        delta = rect_list_obj.height

    unique_values = np.unique(values)
    min_val = np.min(unique_values)
    max_val = np.max(unique_values)
    min_val_rect = np.where(values == min_val)[0]
    max_val_rect = np.where(values == max_val)[0]
    min_list = []
    max_list = []

    for e in range(len(min_val_rect)):
        tmp1_rect = deepcopy(rect_list[min_val_rect[e]])
        tmp2_rect = deepcopy(rect_list[max_val_rect[e]])

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

def find_edge_rect(rect_list_obj, direction):
    rect_list = rect_list_obj.rect_list
    if direction == 'row':
        values = np.array([rect.row for rect in rect_list])
    elif direction == 'range':
        values = np.array([rect.range for rect in rect_list])

    min_val = np.min(np.unique(values))
    max_val = np.max(np.unique(values))

    min_val_rect = np.where(values == min_val)[0]
    max_val_rect = np.where(values == max_val)[0]

    min_list = []
    max_list = []

    for e in range(len(min_val_rect)):
        min_list.append(deepcopy(rect_list[min_val_rect[e]]))
        max_list.append(deepcopy(rect_list[max_val_rect[e]]))

    return min_list, max_list


def correct_rect_range_row(rect_list_obj):
    rect_list = rect_list_obj.rect_list
    range_list = [rect.range for rect in rect_list]
    row_list = [rect.row for rect in rect_list]
    range_list = np.array(range_list)
    row_list = np.array(row_list)
    unique_range = np.unique(range_list)
    unique_row = np.unique(row_list)
    range_cnt = 1
    ID_cnt = 1
    for range_indx in unique_range:
        row_cnt = 1
        for row_indx in unique_row:
            rect_indx = np.where((range_list == range_indx) & (row_list == row_indx))[0][0]
            rect = rect_list[rect_indx]
            rect.range = range_cnt
            rect.row = row_cnt
            rect.ID = ID_cnt
            rect_list[rect_indx] = rect
            ID_cnt += 1
            row_cnt += 1

        range_cnt += 1

    rect_list_obj.rect_list = rect_list
    return rect_list_obj

