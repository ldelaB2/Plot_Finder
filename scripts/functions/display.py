import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib
matplotlib.use('Qt5Agg')
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from functions.optimization import compute_score
from functions.general import bindvec

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

def disp_quadratic_optimization(rect, param_dict, model):
    loss = param_dict['optimization_loss']
    test_points = param_dict['test_points']
    x_radi = param_dict['x_radi']
    y_radi = param_dict['y_radi']
    theta_radi = param_dict['theta_radi']

    # Coompute the current objective function
    current_img = rect.create_sub_image()
    current_img = bindvec(current_img)
    current_score = compute_score(current_img, model, method = loss)
    
    # Compute the objective function
    obj_val = []
    for point in test_points:
        new_img = rect.move_rectangle(point[0], point[1], point[2])
        new_img = bindvec(new_img)
        tmp_score = compute_score(new_img, model, method = loss)
        obj_val.append(tmp_score)

    test_points = np.array(test_points)
    
    # Create the model
    pred_model = make_pipeline(PolynomialFeatures(degree = 2), RandomForestRegressor())
    pred_model.fit(test_points, obj_val)
    
    # Create the meshgrid
    x_disp = np.arange(-x_radi, x_radi, 2)
    y_disp = np.arange(-y_radi, y_radi, 2)
    t_disp = np.arange(-theta_radi, theta_radi, 1)
    X, Y, T = np.meshgrid(x_disp, y_disp, t_disp)
    disp_points = np.column_stack((X.ravel(), Y.ravel(), T.ravel()))
    disp_obj = pred_model.predict(disp_points)

    disp_true = []
    for point in disp_points:
        new_img = rect.move_rectangle(point[0], point[1], point[2])
        new_img = bindvec(new_img)
        tmp_score = compute_score(new_img, model, method = loss)
        disp_true.append(tmp_score)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(disp_points[:,0], disp_points[:,1], disp_true, c = 'r', label = 'True Values')
    ax.scatter(disp_points[:,0], disp_points[:,1], disp_obj, c = 'b', label = 'Predicted Values')
    ax.scatter(test_points[:,0], test_points[:,1], obj_val, c = 'g', label = 'Training Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Objective Function')
    plt.legend()
    plt.show()

    