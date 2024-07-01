import numpy as np
import cv2 as cv
from PIL import Image

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

def dialate_skel(skel):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
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

# Optionally, visualize the flow
def disp_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    vis = img.copy()
    for (x1, y1), (x2, y2) in lines:
        cv.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis