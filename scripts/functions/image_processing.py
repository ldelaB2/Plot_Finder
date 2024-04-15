import numpy as np
import cv2 as cv
from functions.general import find_mode

def create_unit_square(width, height):
    # Creating the unit square
    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    X, Y = np.meshgrid(x, y)
    unit_sqr = np.column_stack((X.ravel(), Y.ravel(), np.ones_like(X.ravel())))

    return unit_sqr 

def create_affine_frame(center_x, center_y, theta, width, height):
    if width != 0:
        width = np.array(width / 2).astype(int)
    if height != 0:
        height = np.array(height / 2).astype(int)

    # Translation Matrix
    t_mat = np.zeros((3, 3))
    t_mat[0, 0], t_mat[1, 1], t_mat[2, 2] = 1, 1, 1
    t_mat[0, 2], t_mat[1, 2] = center_x, center_y

    # Scaler Matrix
    s_mat = np.zeros((3, 3))
    s_mat[0, 0], s_mat[1, 1], s_mat[2, 2] = width, height, 1

    # Rotation Matrix
    r_1 = [np.cos(theta), np.sin(theta), 0]
    r_2 = [-np.sin(theta), np.cos(theta), 0]
    r_3 = [0, 0, 1]
    r_mat = np.column_stack((r_1, r_2, r_3))

    affine_mat = t_mat @ r_mat @ s_mat
    return affine_mat

def compute_points(center_x, center_y, theta, width, height, unit_sqr, img_shape):
    affine_mat = create_affine_frame(center_x, center_y, theta, width, height)
    rotated_points = np.dot(affine_mat, unit_sqr.T).T
    rotated_points = rotated_points[:,:2].astype(int)

   
    # Checking to make sure points are within the image
    img_height, img_width = img_shape[:2]
    valid_y = (rotated_points[:, 1] >= 0) & (rotated_points[:, 1] < img_height)
    valid_x = (rotated_points[:, 0] >= 0) & (rotated_points[:, 0] < img_width)
    invalid_points = (~(valid_x & valid_y))
    rotated_points[invalid_points, :] = [0,0]

    return rotated_points


def extract_rectangle(center_x, center_y, theta, width, height, unit_sqr, img):
    theta = np.radians(theta)
    points = compute_points(center_x, center_y, theta, width, height, unit_sqr, img.shape)

    if len(img.shape) > 2:
        extracted_img = img[points[:, 1], points[:, 0], :]
        extracted_img = np.reshape(extracted_img, (height, width, img.shape[2]))
    else:
        extracted_img = img[points[:, 1], points[:, 0]]
        extracted_img = np.reshape(extracted_img, (height, width))
        
    return extracted_img

def build_path(img_shape, boxradius, skip):
    """
    This function builds a scatter path on the image.

    Parameters:
        img_shape (tuple): The shape of the image.
        boxradius (tuple): The radius of the box to be used in the scatter path.
        skip (tuple): The number of rows and columns to skip between points in the scatter path.

    Returns:
        path (ndarray): A 2D array of points in the scatter path.
        num_points (int): The number of points in the scatter path.

    It first calculates the start and stop points for the rows and columns.
    Then, it creates arrays of row and column indices with the specified skip between them.
    It uses these arrays to create a meshgrid, which it then reshapes into a 2D array of points.
    Finally, it calculates the number of points in the scatter path.
    """
    str1 = 1 + boxradius[0]
    stp1 = img_shape[0] - boxradius[0]
    str2 = 1 + boxradius[1]
    stp2 = img_shape[1] - boxradius[1]

    y = np.arange(str1, stp1, skip)
    x = np.arange(str2, stp2, skip)

    X, Y = np.meshgrid(x, y)
    path = np.column_stack((X.ravel(), Y.ravel()))
    num_points = path.shape[0]
    
    return path, num_points

def find_correct_sized_obj(img):
    #Finding image object stats
    _, labeled_img, stats, _ = cv.connectedComponentsWithStats(img)
    object_areas = stats[:,4]
    object_areas = object_areas[1:]
    normalized_areas = np.zeros((object_areas.size, object_areas.size))
    for e in range(object_areas.size):
        for k in range(e):
            normalized_areas[e,k] = round(object_areas[e] / object_areas[k])

    normalized_areas = normalized_areas + normalized_areas.T
    sum_count = np.sum(normalized_areas == 1, axis = 1)
    mode_count = find_mode(sum_count[sum_count != 0])
    mode_index = np.where(sum_count == mode_count)[0]
    mu = np.mean(object_areas[mode_index])
    
    correct_indx = np.zeros_like(object_areas)
    for e in range(object_areas.size):
        rel_object_size = round(object_areas[e] / mu)
        if rel_object_size > .9 and rel_object_size < 1.25:
            correct_indx[e] = 1
        else:
            correct_indx[e] = 0
    correct_indx = np.where(correct_indx == 1)[0] + 1
    mask = np.isin(labeled_img, correct_indx)
    filtered_img = np.where(mask, 1,0).astype(np.uint8)

    return filtered_img