import json, os, sys, multiprocessing
import numpy as np
import cv2 as cv
from functions.general import bindvec
from matplotlib import pyplot as plt
import random
from sklearn.preprocessing import PolynomialFeatures
from functions.general import bindvec
from scipy.optimize import dual_annealing
from pyswarm import pso
from pyproj import Transformer, CRS

def set_params(param_path):
    
    # Compute the grayscale weights if needed
    if params["gray_scale_method"] == "AUTO":
        if params["auto_grey_weights"] is None or params["recompute_auto_grey_weights"] == True:
            opt_weights, opt_theta = find_g_weights(img, params)
            params["auto_grey_weights"] = opt_weights

        if params["rotation_angle"] == "AUTO":
            params["rotation_angle"] = opt_theta


    # Save the params
    with open(param_path, 'w') as file:
        json.dump(params, file, indent = 4)
    
   
    return params

def compute_GSD(meta_data, logger):
    # Get the current Coordinate Reference System
    current_crs = meta_data["crs"]
    crs_object = CRS.from_user_input(current_crs)
    
    # Find the GSD in the x and y direction in the original crs
    gsd_x = meta_data["transform"][0]
    gsd_y = meta_data["transform"][4]

    # Find the original lat and lon
    original_lat = meta_data["transform"][5]
    original_lon = meta_data["transform"][2]

    if crs_object.is_geographic:
        logger.info("Geographic Coordinate Reference System detected")

        # Dynamically select the UTM zone based on longitude
        utm_zone = int((original_lon + 180) // 6) + 1
        is_northern = original_lat >= 0
        
        # Use UTM CRS for re-projection
        target_crs = CRS.from_proj4(f"+proj=utm +zone={utm_zone} +datum=WGS84 +{'north' if is_northern else 'south'}")

        # Create the transformer to convert the original crs to the target crs
        transformer = Transformer.from_crs(current_crs, target_crs, always_xy=True)

        # Find the x and y in the new crs
        x1, y1 = transformer.transform(original_lon, original_lat)

        # Calculate the GSD in the x direction
        x2, y2 = transformer.transform(original_lon + gsd_x, original_lat)
        gsd_x_m = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Calculate the GSD in the y direction
        x2, y2 = transformer.transform(original_lon, original_lat + gsd_y)
        gsd_y_m = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        gsd_cm = np.round((gsd_x_m + gsd_y_m) / 2 * 100, 4)

    elif crs_object.is_projected:
        logger.info("Projected Coordinate Reference System detected")

        axis_info = crs_object.axis_info[0]
        units = axis_info.unit_name

        if units == 'metre':
            gsd_cm = np.round((gsd_x + gsd_y) / 2 * 100, 4)
        else:
            logger.critical("Invalid unit detected please convert your image to a coordinate system with units of meters or degrees to use AUTO GSD")
            exit(1)

    return gsd_cm


def compute_gray_weights(params, logger):
    # Pull the required params
    row_spacing_inch = params["row_spacing_in"]
    gsd_cm = params["GSD"]
    poly_feature_deg = params["auto_gray_poly_features_degree"]
    gray_features = params["auto_gray_features"]
    #num_images = params["auto_gray_num_subimages"]
    num_features = 50
    img_shape = params["img_ortho_shape"]
    frequency_width = params["freq_filter_width"]

    # Calculating the expected signal based on the row spacing and GSD in pixels
    row_spacing_cm = row_spacing_inch * 2.54
    signal_pixel = row_spacing_cm / gsd_cm

    # Calculating the expected signal baised on GSD and row spacing in frequency
    expected_signal = np.round(img_shape[1] / signal_pixel).astype(int)
    min_expected_signal = np.round(expected_signal * .75).astype(int)
    max_expected_signal = np.round(expected_signal * 1.25).astype(int)

    # Randomly select the rows and columns to sample from the original image
    x_centers = random.sample(range(0, img_shape[1]), num_features)
    y_centers = random.sample(range(0, img_shape[0]), num_features)

    # Sampeling the image at the selected rows and columns
    hort_img = np.zeros(( num_features, img_shape[1], img_shape[2]))
    for n in range(num_features):
        x = x_centers[n]
        y = y_centers[n]
        hort_img[n,:,:] = params["img_ortho"][y,:,:]

   # Compute the features for the horizontal and vertical images
    hort_features = compute_features(hort_img, gray_features, logger)
         
    # Find the final size of the features
    final_shape = hort_features.shape[:2]
    
    # Reshape the sample images
    n_pixels = hort_features.shape[0] * hort_features.shape[1]
    features = hort_features.reshape((n_pixels, hort_features.shape[2]))

    # Create the polynomial features
    poly = PolynomialFeatures(poly_feature_deg)
    poly_features = poly.fit_transform(features)

    # Define the center
    center_x = final_shape[1] // 2

    # Create the signal filter
    sig_filter = np.zeros(final_shape)
    sig_filter[:,(center_x - max_expected_signal):(center_x - min_expected_signal)] = 1
    sig_filter[:,(center_x + min_expected_signal):(center_x + max_expected_signal)] = 1

    sig_indices = np.where(sig_filter == 1)
    noise_indices = np.where(sig_filter == 0)

    def objective_function(args):
        # apply the weights to the polynomial features
        weights = args

        # Apply the weights to the polynomial features
        gray_img = poly_features @ weights
        # Reshape the grayscale image
        gray_img = gray_img.reshape(final_shape)
        # Normalize the grayscale image
        gray_img = (bindvec(gray_img) * 255).astype(np.uint8)

        # Compute the 2D fft of the image
        gray_img = gray_img - np.mean(gray_img)
        two_d_fft = np.fft.fft2(gray_img)
        two_d_fft = np.fft.fftshift(two_d_fft)
        amp = np.abs(two_d_fft)
        amp = np.log(amp + 1)

        sig = np.mean(amp[sig_indices])
        noise = np.mean(amp[noise_indices])

        psnr = 10 * np.log10(sig**2 / noise**2)

        return -psnr

    # Optimize the objective function
    lb = np.array([-1] * (poly_features.shape[1]))
    ub = np.array([1] * (poly_features.shape[1]))
    f_opt, x_opt = pso(objective_function, lb, ub, swarmsize = 50, maxiter = 50, minfunc = 1e-4, omega = 1.5, phip = 0.5, phig = 0.5)
    #bounds = list(zip(lb, ub))
    #result = dual_annealing(objective_function, bounds, maxiter = 100)

    #x_opt = result.x
    #f_opt = result.fun

    # Transform the original image
    gray_img = params["img_ortho"].copy()
    n_pixels = gray_img.shape[0] * gray_img.shape[1]
    gray_img = gray_img.reshape((n_pixels, gray_img.shape[2]))
    gray_img = poly.fit_transform(gray_img)
    gray_img = gray_img @ f_opt
    gray_img = gray_img.reshape(img_shape[:2])
    gray_img = (bindvec(gray_img) * 255).astype(np.uint8)

    

    print(f"Optimal Weights Found! PSNR: {np.round(fopt,2)}")
    print(f"Optimal Theta = {opt_theta}")
    print(f"Optimal Weights for Degree {poly_feature_deg} = {opt_weights}")

    # Create the grayscale image
    return opt_weights, opt_theta





def compute_features(img, gray_features, logger):
    features = []
    for feature in gray_features:
        features.append(compute_gray(False, feature, img, False, logger))

    features = np.array(features)
    features = features.transpose(1,2,0)

    return features

def compute_gray(custom, method, image, invert, logger):
    # convert the img to float32 and read in the grayscale method
    img = np.copy(image).astype(np.float32)

    if custom:
        # Check if the custom method is specified
        eval_expression = method.replace('R', 'img[:,:,0]').replace('G', 'img[:,:,1]').replace('B', 'img[:,:,2]')

        # Use numpy to evaluate the operation
        try:
            pixel_mat = eval(eval_expression)
        except Exception as e:
            logger.critical(f"Error evaluating custom grayscale method: {e}")
            exit(1)

        # Handle division by zero
        pixel_mat = np.where(np.isfinite(pixel_mat), pixel_mat, 0)

    else:
        if method == 'AUTO':
            print("test")


        elif method == 'BI':
            pixel_mat = np.sqrt((img[:,:,0] ** 2 + img[:,:,1] ** 2 + img[:,:,2] ** 2)/3)

        elif method == 'SCI':
            numerator = img[:,:,0] - img[:,:,1]
            denominator = img[:,:,0] + img[:,:,1]
            valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
            pixel_mat = np.where(valid, numerator / denominator, 0)
        
        elif method == 'GLI':
            numerator = 2 * img[:,:,1] - img[:,:,0] - img[:,:,2]
            denominator = 2 * img[:,:,0] + img[:,:,1] + img[:,:,2]
            valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
            pixel_mat = np.where(valid, numerator / denominator, 0)

        elif method == 'HI':
            numerator = 2 * img[:,:,0] - img[:,:,1] - img[:,:,2]
            denominator = img[:,:,1] - img[:,:,2]
            valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
            pixel_mat = np.where(valid, numerator / denominator, 0)

        elif method == 'NGRDI':
            numerator = img[:,:,1] - img[:,:,0]
            denominator = img[:,:,1] + img[:,:,0]
            valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
            pixel_mat = np.where(valid, numerator / denominator, 0)

        elif method == 'SI':
            numerator = img[:,:,0] - img[:,:,2]
            denominator = img[:,:,0] + img[:,:,2]
            valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
            pixel_mat = np.where(valid, numerator / denominator, 0)

        elif method == 'VARI':
            numerator = img[:,:,1] - img[:,:,0]
            denominator = img[:,:,1] + img[:,:,0] - img[:,:,2]
            valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
            pixel_mat = np.where(valid, numerator / denominator, 0)

        elif method == 'BGI':
            numerator = img[:,:,2]
            denominator = img[:,:,1]
            valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
            pixel_mat = np.where(valid, numerator / denominator, 0)

        elif method == 'GRAY':
            pixel_mat = cv.cvtColor(img.astype(np.uint8), cv.COLOR_RGB2GRAY)

        elif method == 'LAB':
            pixel_mat =cv.cvtColor(img.astype(np.uint8), cv.COLOR_RGB2LAB)[:,:,1]

        elif method == 'HSV':
            pixel_mat = cv.cvtColor(img.astype(np.uint8), cv.COLOR_RGB2HSV)[:,:,0]

        else:
            logger.critical(f"Invalid grayscale method: {method}")
            exit(1)

    # Normalize the pixel matrix
    pixel_mat = bindvec(pixel_mat)
    pixel_mat = np.round(pixel_mat * 255).astype(np.uint8)

    # Invert the image if needed
    if invert:
        pixel_mat = 255 - pixel_mat

    return pixel_mat

def rotate_img(img, theta):
    # Computing params for the inverse rotation matrix
    height, width = img.shape[:2]
    rotation_matrix = cv.getRotationMatrix2D((width/2,height/2), theta, 1)
    
    # Determine the size of the rotated image
    cos_theta = np.abs(rotation_matrix[0, 0])
    sin_theta = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin_theta) + (width * cos_theta))
    new_height = int((height * cos_theta) + (width * sin_theta))

    # Adjust the translation in the rotation matrix to prevent cropping
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2

    # Creating the inverse rotation matrix
    inverse_rotation_matrix = cv.getRotationMatrix2D((new_width/2,new_height/2), -theta, 1)
    inverse_rotation_matrix[0, 2] += (width - new_width) / 2
    inverse_rotation_matrix[1, 2] += (height - new_height) / 2
    
    # Rotate the image
    if len(img.shape) == 2:
        rotated_img = cv.warpAffine(img, rotation_matrix, (new_width,new_height), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue = 0)
    else:
        rotated_img = cv.warpAffine(img, rotation_matrix, (new_width,new_height), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue = np.zeros(img.shape[2]))
 
    # Return the inverse rotation matrix, the rotated g image, and the rotated rgb image
    return inverse_rotation_matrix, rotated_img

