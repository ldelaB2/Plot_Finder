import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import multiprocessing, os, json



def set_params(raw_path):
    default_params = {
    "input_path": None, # Path to read input
    "output_path": None, # Path to save output
    "num_cores": None, # Number of cores to use
    "gray_scale_method": "LAB",
    "QC_depth": "min", # Specifies how much of the qc output to save
    "box_radius": [800,500], # Box radius is the size of sub images; 0 = height, 1 = width
    "sparse_skip": [100,100], # Step size for sparse grid
    "freq_filter_width": 1, # Controls how many frequencies we let in when searching
    "row_sig_remove": None, # How many frequencies around the center to set to 0
    "num_sig_returned": 2, # How many frequencies to include in the mask
    "expand_radi": 5, # How many pixels to return for each subI 0 = row, 1 = column
    "wave_pixel_expand": 0, # Controls how many positions in the wave are measured to find pixel value
    "poly_deg_range": 3, # Degree of polynomial used to fit range points
    "poly_deg_row": 1, # Degree of polynomial used to fit row points
    "nrows": None, # Number of rows in the image
    "nranges": None, # Number of ranges in the image
    "optomize_plots": True, # If true, the plot positions are optomized
    "optomization_meta_miter": 5, # Number of times to run the optomization over model
    "optomization_miter": 100, # Number of times to run the optomization over each plot
    "optomization_x_radi": 20, # How many pixels to move the plot in the x direction
    "optomization_y_radi": 50, # How many pixels to move the plot in the y direction
    "optomization_theta_radi": 5, # How many degrees to rotate the plot
    "optomization_import_model": False, # If true, the model is imported from the specified path
    "optomization_model_path": None, # Path to import model from
    "save_plots": True, # If true, the plots are saved in the output Output/Plots
    "create_shapefile": True # If true, a shapefile is created in the output Output/Shapefiles
    }

    user_param_path = os.path.join(raw_path, "params.json")

    try:
        with open(user_param_path, 'r') as file:
            user_params = json.load(file)
    except FileNotFoundError:
        print("No user params found, please create a params.json file")
        exit()
    except json.decoder.JSONDecodeError:
        print("Invalid JSON format, please fix the params.json file")
        exit()
    except:
        print("Unexpected error:", sys.exc_info()[0])
    
    for param, default_value in default_params.items():
        if param not in user_params:
            user_params[param] = default_value
    
    user_params["input_path"] = os.path.join(raw_path, 'Images')
    user_params["output_path"] = os.path.join(raw_path, 'Output')

    try:
        os.mkdir(user_params["output_path"])
    except:
        pass
    
    if user_params["num_cores"] is None:
        user_params["num_cores"] = multiprocessing.cpu_count()
        if user_params["num_cores"] is None:
            user_params["num_cores"] = os.cpu_count()
            if user_params["num_cores"] is None:
                user_params["num_cores"] = 1
    
    if user_params["nrows"] is None:
        print("nrows not specified in params.json")
        exit()
    if user_params["nranges"] is None:
        print("nranges not specified in params.json")
        exit()
    
    return user_params


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

    y = np.arange(str1, stp1, skip[0])
    x = np.arange(str2, stp2, skip[1])

    X, Y = np.meshgrid(x, y)
    path = np.column_stack((X.ravel(), Y.ravel()))
    num_points = path.shape[0]
    
    return path, num_points

def build_rectangles(range_skel, col_skel):
    """
    This function builds rectangles around the connected components in a skeleton image.

    Parameters:
        range_skel (ndarray): The range skeleton image.
        col_skel (ndarray): The column skeleton image.

    Returns:
        list: A list of rectangles. Each rectangle is represented by a tuple containing the center point, width, height, and angle.

    The function first dilates and labels the connected components in the range skeleton image.
    It then creates a boolean mask of the range and column skeleton images and finds the intersection of these masks.
    For each connected component in the range skeleton, it finds the intersection of the component with the intersection mask and sorts these points.
    It then finds the four corners of the bounding rectangle for each pair of points in the sorted list and converts this rectangle into a format with a center point, width, height, and angle.
    Finally, it returns a list of these rectangles.
    """
    def dialated_labeled_skel(skel):
        """
        This function dilates a skeleton image and labels the connected components.

        Parameters:
            skel (ndarray): The input skeleton image.

        Returns:
            tuple: A tuple containing the following elements:
                num_obj (int): The number of connected components in the dilated skeleton.
                labeled_skel (ndarray): The labeled dilated skeleton.
                stats (ndarray): Statistics of the labeled components.
                centroids (ndarray): Centroids of the labeled components.

        The function first creates a structuring element of size (20, 20) and shape ellipse.
        It then dilates the skeleton image using this structuring element.
        It labels the connected components in the dilated skeleton and calculates their statistics and centroids.
        Finally, it returns the number of connected components, the labeled dilated skeleton, the statistics, and the centroids.
        """
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
        dialated_skel = cv.dilate(skel, kernel)
        num_obj, labeled_skel, stats, centroids = cv.connectedComponentsWithStats(dialated_skel)
        return num_obj, labeled_skel, stats, centroids
    
    def four_2_five_rect(points):
        """
        This function converts a rectangle defined by four points into a rectangle defined by a center point, width, height, and angle.

        Parameters:
            points (list): A list of four points defining the rectangle. 
                        The points are in the order: top left, top right, bottom left, bottom right.

        Returns:
            tuple: A tuple containing the center point, width, height, and angle (set to 0) of the rectangle.

        The function first calculates the widths and heights of the rectangle using the Euclidean distance between the corresponding points.
        It then calculates the center point of the rectangle by taking the mean of the four points.
        Finally, it returns a tuple containing the center point, the average width, the average height, and an angle of 0.
        """
        top_left, top_right, bottom_left, bottom_right = points
        w1 = np.linalg.norm(top_left - top_right)
        w2 = np.linalg.norm(bottom_left - bottom_right)
        h1 = np.linalg.norm(top_left - bottom_left)
        h2 = np.linalg.norm(top_right - bottom_right)
        width = ((w1 + w2)/2).astype(int)
        height = ((h1 + h2)/2).astype(int)
        center = np.mean((top_left,top_right,bottom_left,bottom_right), axis = 0).astype(int)
        rect = np.append(center, [width, height, 0])
        return rect

    num_ranges, ld_range, _, _ = dialated_labeled_skel(range_skel)
    
    range_bool = range_skel.astype(bool)
    col_bool = col_skel.astype(bool)
    cp = set(map(tuple, np.argwhere(col_bool & range_bool)))
    rect_list = []
    range_cnt = 0
    rect_cnt = 0

    for e in range(1, num_ranges - 1):
        top_indx = set(map(tuple, np.argwhere(ld_range == e)))
        bottom_indx = set(map(tuple, np.argwhere(ld_range == e + 1)))
        top_points = np.array(list(cp.intersection(top_indx)))
        bottom_points = np.array(list(cp.intersection(bottom_indx)))
        top_points = top_points[np.argsort(top_points[:,1])]
        bottom_points = bottom_points[np.argsort(bottom_points[:,1])]
        row_cnt = 0
        for k in range(top_points.shape[0] - 1):
            top_left = top_points[k,:]
            top_right = top_points[k + 1,:]
            bottom_left = bottom_points[k,:]
            bottom_right = bottom_points[k + 1,:]
            points = [top_left, top_right, bottom_left, bottom_right]
            rect = four_2_five_rect(points)
            position = [range_cnt, row_cnt]
            rect = np.append(rect, position)
            rect_list.append(rect)
            row_cnt += 1
            rect_cnt += 1

        range_cnt += 1

    return rect_list, range_cnt, row_cnt
        
def compute_fft_mat(rect_list, img):
    scores = np.zeros((len(rect_list), rect_list[0].width))
    for e,rect in enumerate(rect_list):
        fsig = rect.compute_fft(img)
        scores[e,:] = fsig

    return scores

def compute_fft_distance(test_set, train_set):
    total_dist = 0
    for e in range(test_set.shape[0]):
        temp_mat = np.zeros_like(train_set)
        temp_mat[:,:] = test_set[e,:]
        raw_dist = temp_mat - train_set
        dist = np.linalg.norm(raw_dist)
        total_dist += dist
    
    return total_dist


def compute_model(rect_list, img):
    width = rect_list[0].width
    height = rect_list[0].height

    model = np.zeros((height, width, 3))
    for rect in rect_list:
        sub_img = rect.create_sub_image(img)
        model = model + sub_img

    model = (model / len(rect_list)).astype(int)
    return model

def disp_rectangles(rect_list, img):
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for rect in rect_list:
        width = (rect.width / 2).astype(int)
        height = (rect.height / 2).astype(int)
        bottom_left_x = rect.center_x - width
        bottom_left_y = rect.center_y - height
        rect_path = patches.Rectangle((bottom_left_x,bottom_left_y), rect.width, rect.height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect_path)

    plt.show()
    return fig, ax

def create_phase2_mask(signal, numfreq):
    ssig = np.argsort(signal)[::-1]
    freq_index = ssig[:numfreq]
    mask = np.zeros_like(signal)
    mask[freq_index] = 1
    
    return mask

def plot_mask(mask):
    plt.ioff()
    plt.close('all')
    fig, ax = plt.subplots()
    ax.plot(mask)
    offset = 0
    for i, val in enumerate(mask):
        if val > 0:
            plt.annotate(str(i), (i, val + offset))
            offset -= .1

    return fig, ax

def find_mode(arr):
    """
    This function finds the mode of a given array.

    Parameters:
        arr (ndarray): The input array.

    Returns:
        int or float: The mode of the array.

    The function first uses numpy's unique function to find the unique elements in the array and their counts.
    It then finds the index of the maximum count, which corresponds to the mode of the array.
    If there are multiple modes, it prints a message and returns the first mode.
    """
    unique_elements, counts = np.unique(arr, return_counts = True)
    max_count_index = np.argmax(counts)
    modes = unique_elements[counts == counts[max_count_index]]

    if len(modes) == 1:
        return modes[0]
    else:
        print("Multiple Modes found returning first")
        return modes[0]


def findmaxpeak(signal, mask = None):
    """
    This function finds the index of the maximum peak in a signal.

    Parameters:
        signal (ndarray): The input signal.
        mask (ndarray, optional): An optional mask to apply to the signal before finding the peak. 
                                  If not provided, a mask of ones (i.e., no change) is used.

    Returns:
        int: The index of the maximum peak in the signal.

    The function first checks if a mask is provided. If not, it creates a mask of ones with the same shape as the signal.
    It then applies the mask to the signal by element-wise multiplication.
    Finally, it finds and returns the index of the maximum value in the masked signal.
    """
    if mask is None:
        mask = np.ones_like(signal)

    signal = signal * mask
    out = np.argmax(signal)
    return out


def find_center_line(args):
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


def bindvec(in_array):
    """
    This function normalizes an input array.

    Parameters:
        in_array (ndarray): The input array.

    Returns:
        ndarray: The normalized array.

    The function first subtracts the minimum value of the array from all elements of the array.
    It then divides all elements of the array by the maximum value of the array to normalize it to the range [0, 1].
    Finally, it reshapes the array back to its original shape and returns it.
    """
    z = in_array.shape
    in_array = in_array.ravel() - np.min(in_array)
    max_val = np.max(in_array)
    if max_val == 0:
        out = in_array
    else:
        out = in_array * (1.0 / max_val)
    out = out.reshape(z)
    return out


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

