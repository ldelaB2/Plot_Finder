import numpy as np
import cv2 as cv
from functions.general import bindvec
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from functions.general import bindvec
from pyproj import Transformer, CRS
from functions.image_processing import build_path
from classes.sub_image import sub_image
from matplotlib import pyplot as plt

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
    img_size = user_params["img_ortho_shape"]
    gsd = user_params["GSD"]
    row_spacing_cm = user_params["row_spacing_in"] * 2.54
    range_spacing_cm = user_params["row_length_ft"] * 30.48

    signal_pixel_row = np.round(row_spacing_cm / gsd).astype(int)
    signal_pixel_range = np.round(range_spacing_cm / gsd).astype(int)

    target_row_signal = 12
    target_range_signal = 5

    desired_box_width = target_row_signal * signal_pixel_row // 2
    desired_box_height = target_range_signal * signal_pixel_range // 2
    
    if (2 * desired_box_width + 1) > img_size[1]:
        logger.warning(f"Row signal is too small. Using Full Image Width")
        box_width_radi = img_size[1] // 2 - 1
        row_sig_remove = 0
    else:
        box_width_radi = desired_box_width
        row_sig_remove = 4
        logger.info(f"Using box width of {box_width_radi}")

    if (2 * desired_box_height + 1) > img_size[0]:
        logger.warning(f"Range signal is too small. Using Full Image Height")
        box_height_radi = img_size[0] // 2 - 1
        range_sig_remove = 0
    else:
        box_height_radi = desired_box_height
        logger.info(f"Using box height of {box_height_radi}")
        range_sig_remove = 2

    logger.info(f"Frequency Supression: Row: {row_sig_remove}, Range: {range_sig_remove}")

    box_radi = (box_height_radi, box_width_radi)
    frequency_supression = (row_sig_remove, range_sig_remove)

    return box_radi, frequency_supression

def compute_skip(img_size, box_radi, num_images, logger):
    valid_cols = np.array([1 + box_radi[1], img_size[1] - box_radi[1]])
    valid_rows = np.array([1 + box_radi[0], img_size[0] - box_radi[0]])

    col_range = valid_cols[1] - valid_cols[0]
    row_range = valid_rows[1] - valid_rows[0]
    
    ratio = col_range / row_range

    num_col_samples = int(np.sqrt(num_images * ratio))
    num_row_samples = int(np.sqrt(num_images / ratio))

    col_spacing = int(col_range / num_col_samples)
    row_spacing = int(row_range / num_row_samples)

    if col_spacing < 1:
        col_spacing = 1
        logger.warning("Col spacing is too small. Using 1")
    else:
        logger.info(f"Col Spacing: {col_spacing}")
    if row_spacing < 1:
        row_spacing = 1
        logger.warning("Row spacing is too small. Using 1")
    else:
        logger.info(f"Row Spacing: {row_spacing}")

    
    skip = (row_spacing, col_spacing)
    return skip

def compute_signal(user_params, logger):
    # Pulling the params
    img_shape = user_params["img_ortho_shape"]
    box_radi = user_params["box_radi"]
    frequency_supression = user_params["frequency_supression"]
    sparse_skip_radi = user_params["sparse_skip_radi"]
    sparse_skip_num_images = user_params["sparse_skip_num_images"]
    img = user_params["gray_img"]
    freq_filter_width = user_params["freq_filter_width"]
    num_sig_returned = user_params["num_sig_returned"]
    gsd = user_params["GSD"]

    # Check if the box_radi and frequency_supression are None
    if box_radi is None or frequency_supression is None:
        logger.info("Computing box size")
        box_radi, frequency_supression = compute_box_size(user_params, logger)

        # Update the user params
        user_params["box_radi"] = box_radi
        user_params["frequency_supression"] = frequency_supression
    else:
        logger.info("Using user specified box size")

    # Check if the sparse_skip_radi is None
    if sparse_skip_radi is None:
        logger.info(f"Computing sparse skip using {sparse_skip_num_images} images")
        sparse_skip_radi = compute_skip(img_shape, box_radi, sparse_skip_num_images, logger)
        
        # Update the user params
        user_params["sparse_skip_radi"] = sparse_skip_radi
    else:
        logger.info("Using user specified sparse skip")
    
    sparse_grid, num_points = build_path(img_shape, box_radi, sparse_skip_radi)

    # Preallocate memory
    range_waves = np.zeros((num_points, (2 * box_radi[0])))
    row_waves = np.zeros((num_points, (2 * box_radi[1])))

    logger.info(f"Computing the signal using {num_points} points")

    for e in range(num_points):
        subI = sub_image(img, box_radi, sparse_grid[e])
        row_waves[e, :], range_waves[e, :] = subI.phase1(freq_filter_width)

    # Compute the dominant frequency in the row and range direction
    row_sig = np.mean(row_waves, 0)
    range_sig = np.mean(range_waves, 0)

    # Set frequencies to zero if they are in the frequency supression range
    row_sig_remove = np.arange((len(row_sig) // 2 - frequency_supression[0]),(len(row_sig) // 2 + frequency_supression[0]),1)
    range_sig_remove = np.arange((len(range_sig) // 2 - frequency_supression[1]),(len(range_sig) // 2 + frequency_supression[1]),1)

    row_sig[row_sig_remove] = 0
    range_sig[range_sig_remove] = 0

    max_row_sig = np.argsort(row_sig)[::-1][:num_sig_returned]
    max_range_sig = np.argsort(range_sig)[::-1][:num_sig_returned]

    implied_row_spacing = []
    implied_range_spacing = []
    for e in range(num_sig_returned):
        implied_row_spacing.append((len(row_sig) / abs(max_row_sig[e] - len(row_sig) // 2)) * gsd)
        implied_range_spacing.append((len(range_sig) / abs(max_range_sig[e] - len(range_sig) // 2)) * gsd)

    # Result in in cm convert to inch and feet
    implied_row_spacing = np.round(np.mean(implied_row_spacing) * .393701, 1)
    implied_range_spacing = np.round(np.mean(implied_range_spacing) * .0328084, 1)

    logger.info(f"Implied Row Spacing: {implied_row_spacing} in")
    logger.info(f"Implied Range Spacing: {implied_range_spacing} ft")
    logger.info("Finished Processing Sparse Grid")

    return max_row_sig, max_range_sig




