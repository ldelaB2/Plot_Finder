import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv

def break_up_skel(args):
    kernel, skel, shape = args
    kernel = np.array(kernel).reshape(shape[0], shape[1]).astype(np.uint8)
    tmp = np.argwhere(cv.erode(skel.astype(np.uint8), kernel, iterations=1).astype(bool))
    if tmp.size > 0:
        tmp = tmp.tolist()
        return tmp
    else:
        return




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

