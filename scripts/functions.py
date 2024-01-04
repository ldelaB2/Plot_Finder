import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

def build_path(img_shape, boxradius, skip):
    str1 = 1 + boxradius[0]
    stp1 = img_shape[0] - boxradius[0]
    str2 = 1 + boxradius[1]
    stp2 = img_shape[1] - boxradius[1]

    y = np.arange(str1, stp1, skip[0])
    x = np.arange(str2, stp2, skip[1])

    X, Y = np.meshgrid(x, y)
    path = np.column_stack((X.ravel(), Y.ravel()))
    num_points = path.shape[0]
    
    return path, num_points

def find_points(start_point, points, point_list, flag):
    next_point = eludian_distance(start_point, points, True)[0, :]
    if next_point[1] < start_point[1]:
        return
    else:
        point_list.append(next_point)
        indx = np.where((points[:, 0] == next_point[0]) & (points[:, 1] == next_point[1]))
        points = np.delete(points, indx, axis=0)
        if points.shape[0] == 1:
            point_list.append(points[0,:])
            flag[0] = True
            return
        else:
            find_points(next_point, points, point_list, flag)


def eludian_distance(target, points, return_points=False):
    target = np.array(target)
    points = np.array(points)
    dist = np.sqrt(np.sum((target - points) ** 2, axis=1))
    if return_points:
        indx = np.argsort(dist)
        points = points[indx]
        return points
    else:
        return dist

def create_phase2_mask( signal, numfreq, radius=None, supressor=None, disp=False):
    if supressor is not None:
        signal[(radius - supressor):(radius + supressor)] = 0
        ssig = np.argsort(signal)[::-1]
    else:
        ssig = np.argsort(signal)[::-1]

    freq_index = ssig[:numfreq]
    mask = np.zeros_like(signal)
    mask[freq_index] = 1

    if disp:
        plt.plot(mask)
        plt.show()
    return mask

# Find the mode of a vector
def find_mode(arr):
    unique_elements, counts = np.unique(arr, return_counts = True)
    max_count_index = np.argmax(counts)
    modes = unique_elements[counts == counts[max_count_index]]

    if len(modes) == 1:
        return modes[0]
    else:
        print("Multiple Modes found returning first")
        return modes[0]

# Find max peak
def findmaxpeak(signal, mask = None):
    if mask is None:
        mask = np.ones_like(signal)

    signal = signal * mask
    out = np.argmax(signal)
    return out

def find_center_line(args):
    obj_labeled_img, indx, poly_degree, img_size = args
    
    subset_y = np.column_stack(np.where(obj_labeled_img == indx))
    # Calculate mean 'x' for each 'y'
    unique_vales, counts = np.unique(subset_y[:, 0], return_counts=True)
    x_position = np.bincount(subset_y[:, 0], weights=subset_y[:, 1])
    mean_x_values = (x_position[unique_vales] / counts).astype(int)
    # Fit a polynomial to these points
    coefficients = np.polyfit(unique_vales, mean_x_values, poly_degree)
    poly = np.poly1d(coefficients)
    # Get the x and y coordinates
    x = np.arange(0, img_size, 1)
    y = poly(x).astype(int)
    # Create a list of tuples for the centerline
    centerline = list(zip(y, x))

    return centerline

# Normalize a vector
def bindvec(in_array):
    z = in_array.shape
    in_array = in_array.ravel() - np.min(in_array)
    out = in_array * (1.0 / np.max(in_array))
    out = out.reshape(z)
    return out

def flatten_mask_overlay(image, mask, alpha = 0.5):
    color = [1, 0, 0]
    img_copy = np.copy(image)

    if np.ndim(image) > 2:
        img_red = img_copy[:,:,0]
        img_green = img_copy[:,:,1]
        img_blue = img_copy[:,:,2]
    else:
        img_red = img_copy
        img_green = img_copy
        img_blue = img_copy

    rMask = color[0] * mask
    gMask = color[1] * mask
    bMask = color[2] * mask

    rOut = img_red + (rMask - img_red) * alpha
    gOut = img_green + (gMask - img_green) * alpha
    bOut = img_blue + (bMask - img_blue) * alpha

    img_red[rMask == 1] = rOut[rMask == 1]
    img_green[gMask == 1] = gOut[gMask == 1]
    img_blue[bMask == 1] = bOut[bMask == 1]

    img_copy = np.stack([img_red,img_green,img_blue], axis = -1).astype(np.uint8)
    img_copy = Image.fromarray(img_copy)
    return(img_copy)

