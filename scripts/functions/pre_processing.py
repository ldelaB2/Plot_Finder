import numpy as np
import cv2 as cv
from functions.general import bindvec
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from functions.general import bindvec
from pyproj import Transformer, CRS

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
    logger.info("Not implemented yet")
    #TODO implement this function
    return

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



def compute_theta(user_params, logger):
    # Read in the image
    img = user_params["img_ortho"]
    
    # Compute the signal pixel
    gsd = user_params["GSD"]
    row_spacing_cm = user_params["row_spacing_in"] * 2.54
    signal_pixel = np.round(row_spacing_cm / gsd).astype(int)

    # Create the signal and noise masks
    signal_mask = np.zeros((1, img.shape[1]))
    sig_index = np.array([img.shape[1] // 2 - signal_pixel, img.shape[1] // 2 + signal_pixel])
    signal_mask[0, sig_index] = 1
    dialated_signal_mask = cv.dilate(signal_mask, np.ones((1, 5), np.uint8), iterations = 3)
    
    # Compute the signal and noise index
    signal_index = np.where(dialated_signal_mask == 1)[1]
    noise_index = np.where(dialated_signal_mask == 0)[1]

    # Compute the gray image
    gray_img = compute_gray(False, "GRAY", img, False, logger)
    
    def compute_theta_psnr(theta):
        # Rotate the image
        rotated_img = cv.warpAffine(gray_img, cv.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), theta, 1), (img.shape[1], img.shape[0]), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue = 0)       

        # compute the signal
        signal = np.mean(rotated_img, axis = 0)
        signal = signal - np.mean(signal)

        fft = np.fft.fft(signal)
        fft = np.fft.fftshift(fft)
        fft = np.abs(fft)

        # Compute the noise
        noise = np.mean(fft[noise_index])
        signal = np.mean(fft[signal_index])

        psnr = 10 * np.log10(signal**2 / noise**2)
        
        return psnr
    
    # Compute the PSNR for each theta
    theta_range = np.arange(-90, 90, 1)
    psnr_list = []

    for theta in theta_range:
        psnr = compute_theta_psnr(theta)
        psnr_list.append(psnr)

    # Find the theta with the highest PSNR
    opt_theta = theta_range[np.argmax(psnr_list)]

    return opt_theta


def compute_box_size(user_params, logger):
    print()
    
def compute_signal(user_params, logger):



