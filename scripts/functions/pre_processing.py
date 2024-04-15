import json, os, sys, multiprocessing
import numpy as np
import cv2 as cv
from functions.general import bindvec
from matplotlib import pyplot as plt

def set_params(param_path):
    # Read in the default params
    default_param_path = os.path.join(os.path.dirname(__file__), "default_params.json")
    with open(default_param_path, 'r') as file:
        default_params = json.load(file)

    try:
        with open(param_path, 'r') as file:
            params = json.load(file)
    except FileNotFoundError:
        print("No user params found, please create a params.json file")
        exit()
    except json.decoder.JSONDecodeError:
        print("Invalid JSON format, please fix the params.json file")
        exit()
    except:
        print("Unexpected error:", sys.exc_info()[0])

    # Check if all the default params are in the user params
    for param, default_value in default_params.items():
        if param not in params:
            params[param] = default_value
    
    if params["img_path"] is None:
        print("img_path not specified in params.json")
        exit()
    if params["output_path"] is None:
        print("output_path not specified in params.json")
        exit()
    if params["nrows"] is None:
        print("nrows not specified in params.json")
        exit()
    if params["nranges"] is None:
        print("nranges not specified in params.json")
        exit()

    # Create the output directory if it does not exist
    try:
        os.mkdir(params["output_path"])
    except:
        pass
    
    # Check if the number of cores is specified
    if params["num_cores"] is None:
        params["num_cores"] = multiprocessing.cpu_count()
        if params["num_cores"] is None:
            params["num_cores"] = os.cpu_count()
            if params["num_cores"] is None:
                params["num_cores"] = 1
    
   
    return params

def create_g(img, params):
    # convert the img to float32 and read in the grayscale method
    img = img.astype(np.float32)
    method = params["gray_scale_method"]

    if params["custom_grayscale"] == True:
        # Check if the custom method is specified
        eval_expression = method.replace('R', 'img[:,:,0]').replace('G', 'img[:,:,1]').replace('B', 'img[:,:,2]')

        # Use numpy to evaluate the operation
        try:
            pixel_mat = eval(eval_expression)
        except Exception as e:
            print(f"Error evaluating custom method: {e}")
            return

        # Handle division by zero
        pixel_mat = np.where(np.isfinite(pixel_mat), pixel_mat, 0)

    else:
        if method == 'BI':
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

        elif method == 'GREY':
            pixel_mat = cv.cvtColor(img.astype(np.uint8), cv.COLOR_RGB2GRAY)

        elif method == 'LAB':
            pixel_mat =cv.cvtColor(img.astype(np.uint8), cv.COLOR_RGB2LAB)[:,:,1]

        elif method == 'HSV':
            pixel_mat = cv.cvtColor(img.astype(np.uint8), cv.COLOR_RGB2HSV)[:,:,0]

        else:
            print("Invalid method")
            exit()

    # Normalize the pixel matrix
    pixel_mat = bindvec(pixel_mat)
    pixel_mat = np.round(pixel_mat * 255).astype(np.uint8)

    # Invert the image if needed
    if params["gray_scale_invert"] == True:
        pixel_mat = 255 - pixel_mat

    return pixel_mat

def create_output_dirs(params):
    subdir = os.path.join(params["output_path"], params["img_name"] + "_pf_output")
    output_paths = {}
    try:
        os.mkdir(subdir)
    except:
        pass

    #Check if extracting plots
    if params["save_plots"] == True:
        plot_dir = os.path.join(subdir, "plots")
        output_paths["plot_dir"] = plot_dir
        try:
            os.mkdir(plot_dir)
        except:
            pass

    # Check if saving QC
    if params["QC_depth"] != "none":
        qc_dir = os.path.join(subdir, "QC")
        output_paths["qc_dir"] = qc_dir
        try:
            os.mkdir(qc_dir)
        except:
            pass

    # Check if saving Shapefiles
    if params["create_shapefile"] == True:
        shape_dir = os.path.join(subdir, "shapefiles")
        output_paths["shape_dir"] = shape_dir
        try:
            os.mkdir(shape_dir)
        except:
            pass

    # Check if exporting optimization models
    if params["optimize_plots"] == True and params["optimization_export_model"] == True:
        opt_dir = os.path.join(subdir, "optimization_models")
        output_paths["opt_dir"] = opt_dir
        try:
            os.mkdir(opt_dir)
        except:
            pass

    return output_paths

def rotate_img(g_img, rgb_img, rotation_angle):
    if rotation_angle is None:
        theta = find_theta(g_img)
    else:
        theta = rotation_angle

    # Computing params for the inverse rotation matrix
    height, width = g_img.shape[:2]
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
    g_img = cv.warpAffine(g_img, rotation_matrix, (new_width,new_height), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue = 0)
    rgb_img = cv.warpAffine(rgb_img, rotation_matrix, (new_width, new_height), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue = (0,0,0))
    
    # Return the inverse rotation matrix, the rotated g image, and the rotated rgb image
    return inverse_rotation_matrix, g_img, rgb_img

def find_theta(g_img):
    # Calculating 2d FFT
    mean_subtracted = g_img - np.mean(g_img)
    rot_fft = np.fft.fft2(mean_subtracted)
    rot_fft = np.fft.fftshift(rot_fft)
    rot_fft = np.abs(rot_fft)

    # Creating the sub image
    box_radi = 30
    x_center = rot_fft.shape[1] // 2
    y_center = rot_fft.shape[0] // 2
    sub_image = rot_fft[y_center - box_radi:y_center + box_radi, x_center - box_radi:x_center + box_radi]
    #f_size = 3
    #sub_image[box_radi - f_size:box_radi + f_size, box_radi - f_size:box_radi + f_size] = 0

    x1,x2 = np.meshgrid(np.arange(-box_radi,box_radi),np.arange(-box_radi,box_radi))
    features = np.stack((x1.ravel(),x2.ravel()),axis = 1)
    response = sub_image.reshape((box_radi * 2) ** 2, 1)
    w = np.linalg.inv(features.T @ features) @ features.T @ response
    theta = np.degrees(np.arctan(w[1] / w[0]))[0]

    x = np.arange(60)
    y = x * np.cos(np.deg2rad(theta))
    y = -y


    return theta
