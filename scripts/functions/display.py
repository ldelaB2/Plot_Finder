import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt
import os

def flatten_mask_overlay(image, mask, alpha = 0.5):
    """
    This function overlays a mask on an image with a specified transparency.

    Parameters:
        image (ndarray): The input image.
        mask (ndarray): The mask to overlay on the image.
        alpha (float, optional): The transparency of the overlay. Defaults to 0.5.

    Returns:
        ndarray: The image with the mask overlay.

    The function first checks if the image is a color image (i.e., has more than 2 dimensions).
    If it is, it separates the red, green, and blue channels of the image.
    If it is not, it uses the grayscale image for all three channels.
    It then multiplies the mask by the color for each channel to create a colored mask.
    Finally, it overlays the colored mask on each channel of the image using the specified transparency and returns the result.
    """

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



def dialate_skel(skel, kernel_size = 20):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dialated_skel = cv.dilate(skel, kernel)

    return dialated_skel

def disp_rectangles(rect_list):
    output_img = np.copy(rect_list[0].img)
    width = (rect_list[0].width / 2).astype(int)
        
    for rect in rect_list:
        points = rect.compute_corner_points()

        if rect.flagged:
            output_img = cv.polylines(output_img, [points], True, (0, 255, 0), 10)
        else:
            output_img = cv.polylines(output_img, [points], True, (255, 0, 0), 10)

        if rect.ID is not None:
            font = cv.FONT_HERSHEY_SIMPLEX
            scale = 1.5
            color = (255,255,255)
            position = (rect.center_x - width, rect.center_y)
            thickness = 5
            txt = str(rect.ID)
            output_img = cv.putText(output_img, txt, position, font, scale, color, thickness, cv.LINE_AA)
        
    output_img = Image.fromarray(output_img)

    return output_img

def disp_spiral_path(path):
    max_range = np.max(path[:,0])
    max_row = np.max(path[:,1])

    fig, ax = plt.subplots()
    scat = ax.scatter([],[])
    ax.set_xlim(0, max_row + 1)
    ax.set_ylim(0, max_range + 1)

    for point in path:
        ax.scatter(point[1], point[0], c = 'b')
        plt.pause(.1)

    plt.close('all')

    return

def disp_distance_change(expected_centers, geometric_mean, current_center):
    expected_centers = np.array(expected_centers)
    geometric_mean = np.array(geometric_mean)
    current_center = np.array(current_center)
    plt.scatter(expected_centers[:,0], expected_centers[:,1], c = 'r', label = 'Expected Center')
    plt.scatter(geometric_mean[0], geometric_mean[1], c = 'g', label = 'Geometric Mean')
    plt.scatter(current_center[0], current_center[1], c = 'b', label = 'Current Center')
    plt.legend()
    plt.show()

def save_results(params, images, name, type, logger):
    flag = params["save_QC"]

    if not flag:
        return
    else:
        output_directory = params["pf_output_directorys"]["quality"]

        if len(images) != len(name):
            logger.info("Number of images and names do not match")
            return

        if type == "image":
            for e in range(len(images)):
                output_path = os.path.join(output_directory, f"{name[e]}.jpg")
                cv.imwrite(output_path, images[e])

        elif type == "wavepad":
            for e in range(len(images)):
                output_path = os.path.join(output_directory, f"{name[e]}.jpg")
                output = flatten_mask_overlay(params["img_ortho"], images[e])
                output.save(output_path)

        elif type == "skel":
            for e in range(len(images)):
                output_path = os.path.join(output_directory, f"{name[e]}.jpg")
                output = flatten_mask_overlay(params["img_ortho"], dialate_skel(images[e]))
                output.save(output_path)

        elif type == "rect_list":
            for e in range(len(images)):
                output_path = os.path.join(output_directory, f"{name[e]}.jpg")
                output = disp_rectangles(images[e])
                output.save(output_path)

        else:
            logger.info("Invalid type for saving results")
            return
        
def save_model(params, model, name, logger):
    flag = params["save_optimization_model"]

    if not flag:
        return
    else:
        output_directory = params["pf_output_directorys"]["optimization_models"]
        output_path = os.path.join(output_directory, f"{name}.jpg")
        cv.imwrite(output_path, model)
        


    