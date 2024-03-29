import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import multiprocessing, os, json
from copy import deepcopy
from scipy.stats import norm, expon, poisson
from scipy.optimize import minimize, curve_fit, dual_annealing



def set_params(param_path):
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
    
    for param, default_value in default_params.items():
        if param not in params:
            params[param] = default_value
    
    if params["input_path"] is None:
        print("input_path not specified in params.json")
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

    try:
        os.mkdir(params["output_path"])
    except:
        pass
    
    if params["num_cores"] is None:
        params["num_cores"] = multiprocessing.cpu_count()
        if params["num_cores"] is None:
            params["num_cores"] = os.cpu_count()
            if params["num_cores"] is None:
                params["num_cores"] = 1
    
   
    return params


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
        
def pca(data, num_comp):
    # Center the data
    data = data - np.mean(data, axis = 0)

    # Scale the data
    col_std = np.std(data, axis = 0)
    if np.any(col_std == 0):
        print("Column with zero standard deviation found")
        exit()
    else:
        data = data / col_std

    # Compute the covariance matrix
    cov = np.cov(data, rowvar = False)

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort the eigenvalues and eigenvectors
    sorted_indx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indx]
    eigenvalues = eigenvalues[sorted_indx]
    eign_pve = eigenvalues / np.sum(eigenvalues)

    # Return the correct number of components
    pc_eign_vect = eigenvectors[:, :num_comp]
    pc_eign_val = eigenvalues[:num_comp]
    pc_eign_pve = eign_pve[:num_comp]

    proj_data = np.dot(data, pc_eign_vect)

    return proj_data, pc_eign_vect, pc_eign_val, pc_eign_pve
    


def four_2_five_rect(points):
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

def dialated_labeled_skel(skel):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
    dialated_skel = cv.dilate(skel, kernel)
    num_obj, labeled_skel, stats, centroids = cv.connectedComponentsWithStats(dialated_skel)

    return num_obj, labeled_skel, stats, centroids


def compute_fft_mat(rect_list, img):
    scores = np.zeros((len(rect_list), rect_list[0].width))
    for e,rect in enumerate(rect_list):
        fsig = rect.compute_fft(img)
        scores[e,:] = fsig

    return scores

def compute_fft_distance(test_set, train_set):
    train_mat = np.tile(train_set, (test_set.shape[0], 1))
    dist_mat = train_mat - test_set
    dist = np.linalg.norm(dist_mat)

    return dist

def filter_wavepad(wavepad, method = "otsu"):
    if method == "otsu":
        _, binary_wavpad = cv.threshold(wavepad, 0, 1, cv.THRESH_OTSU)
        kernel = np.ones((5,5), np.uint8)
        binary_wavpad = cv.erode(binary_wavpad, kernel, iterations = 3)
        return binary_wavpad

    elif method == "hist":
        def hist_filter_wavepad(wavepad):
            def model_function(x, lambda1, lambda2, p):
                left_dist = p * poisson.pmf(x, mu = lambda1) 
                right_dist = (1 - p) * poisson.pmf(255 - x, mu = lambda2)
                total_dist = left_dist + right_dist
                return total_dist
            
            def calc_prob(x, lambda1, lambda2, p):
                left_dist = p * poisson.pmf(x, mu = lambda1) 
                right_dist = (1 - p) * poisson.pmf(255 - x, mu = lambda2)
                total_dist = left_dist + right_dist
               
                left_prob = np.where(total_dist > 0, left_dist / total_dist, 0)
                right_prob = np.where(total_dist > 0, right_dist / total_dist, 0)
                return left_prob, right_prob
            
            def objective_function(params, bin_centers, pixels_hist):
                prob_dist = model_function(bin_centers, *params)
                dist = - np.sum(pixels_hist * np.log(prob_dist + 1e-10))
                return dist
            
            pixels = wavepad.reshape(-1)
            pixels_hist, _ = np.histogram(pixels, bins = 256, density=True)
            bin_left_edge = np.arange(0, 256, 1)

            bounds = [(0,1), (0,1), (0,1)]
            results = dual_annealing(objective_function, bounds, args=(bin_left_edge, pixels_hist), maxiter = 1000)
            opt_params = results.x

            #prob_dist = model_function(bin_left_edge, *opt_params)
            # Compute the probability of each point
            point_left_prob, point_right_prob = calc_prob(pixels, *opt_params)
            binary_wavepad = np.where(point_left_prob >= point_right_prob, 0, 1)

            # Create the wavepad
            binary_wavepad = binary_wavepad.reshape(wavepad.shape)

            return binary_wavepad

        def fft_filter_wavepad(binary_wavepad):
            center = np.round(binary_wavepad.shape[0] / 2).astype(int)
            fft_filter_squash = 2
            fft_amp_wavepad = np.zeros_like(binary_wavepad, dtype = np.float64)
            fft_phase_wavepad = np.zeros_like(binary_wavepad, dtype= np.float64)

            for e in range(binary_wavepad.shape[1]):
                # Computing the fft
                col = binary_wavepad[:,e]
                col = col - np.mean(col)
                fft_col = np.fft.fft(col)
                fft_col = np.fft.fftshift(fft_col)
                fft_amp = np.abs(fft_col)
                fft_phase = np.angle(fft_col)

                # Creating the filter
                fft_filter = np.ones_like(fft_amp)
                fft_filter[center - fft_filter_squash: center + fft_filter_squash] = 0

                # Applying the filter
                fft_amp = fft_amp * fft_filter
                fft_phase = fft_phase * fft_filter
                fft_amp_wavepad[:,e] = fft_amp
                fft_phase_wavepad[:,e] = fft_phase
            
            # Creating the phase 2 mask
            avg_fft = np.mean(fft_amp_wavepad, axis = 1)
            max_indx = np.sort(avg_fft.argsort()[-2:])
            mask = np.zeros_like(avg_fft)
            mask[max_indx] = 1
            
            # Create the wavepad
            wavepad = np.zeros_like(binary_wavepad, dtype = np.float64)
            for e in range(binary_wavepad.shape[1]):
                # Computing the spacial wave
                fft_amp = fft_amp_wavepad[:,e] * mask
                fft_phi = fft_phase_wavepad[:,e] * mask
                fft_freq_wave = fft_amp * np.exp(1j * fft_phi)
                spacial_wave = np.fft.ifft(np.fft.fftshift(fft_freq_wave))
                #spacial_wave = np.round(bindvec(np.real(spacial_wave)) * 255).astype(np.uint8)
                wavepad[:,e] = np.real(spacial_wave)

            # Fix type and return
            wavepad = np.round(bindvec(wavepad) * 255).astype(np.uint8)
            return wavepad

        epoc = 2
        for e in range(epoc):
            wavepad = hist_filter_wavepad(wavepad)
            wavepad = fft_filter_wavepad(wavepad)
            if e == epoc - 1:
                wavepad = hist_filter_wavepad(wavepad)

        return wavepad.astype(np.uint8)
    
    else:
        print("Invalid filtering method")
        exit()


def find_correct_sized_obj(img):
    #Finding image object stats
    _, labeled_img, stats, _ = cv.connectedComponentsWithStats(img)
    object_areas = stats[:,4]
    object_areas = object_areas[1:]
    normalized_areas = np.zeros((object_areas.size, object_areas.size))
    for e in range(object_areas.size):
        for k in range(e):
            normalized_areas[e,k] = round(object_areas[e] / object_areas[k])

    normalized_areas = normalized_areas + normalized_areas.T
    sum_count = np.sum(normalized_areas == 1, axis = 1)
    mode_count = find_mode(sum_count[sum_count != 0])
    mode_index = np.where(sum_count == mode_count)[0]
    mu = np.mean(object_areas[mode_index])
    
    correct_indx = np.zeros_like(object_areas)
    for e in range(object_areas.size):
        rel_object_size = round(object_areas[e] / mu)
        if rel_object_size > .9 and rel_object_size < 1.25:
            correct_indx[e] = 1
        else:
            correct_indx[e] = 0
    correct_indx = np.where(correct_indx == 1)[0] + 1
    mask = np.isin(labeled_img, correct_indx)
    filtered_img = np.where(mask, 1,0).astype(np.uint8)

    return filtered_img

def filter_rectangles(rect_list, img):
    # Computing the model
    avg_hist = np.zeros((3,256))
    for rect in rect_list:
        t_red, t_green, t_blue = rect.compute_histogram(img)
        avg_hist[0, :] +=  t_red[0]
        avg_hist[1, :] += t_green[0]
        avg_hist[2, :] += t_blue[0]
    
    avg_hist = (avg_hist / len(rect_list)).astype(int)

    # Computing the distance
    hist_dist = []
    for rect in rect_list:
        t_red, t_green, t_blue = rect.compute_histogram(img)
        t_hist = np.vstack((t_red[0], t_green[0], t_blue[0]))
        dist = np.linalg.norm(t_hist - avg_hist).astype(int)
        hist_dist.append(dist)
    
    hist_dist = np.array(hist_dist)
    mean_dist = np.mean(hist_dist)
    std_dist = np.std(hist_dist)
    threshold = mean_dist  + 3 * std_dist
    flagged_indx = np.where(hist_dist > threshold)[0]
    model_indx = np.where(hist_dist <= mean_dist)[0]

    flagged_rect_list = [rect_list[indx] for indx in flagged_indx]
    for rect in flagged_rect_list:
        rect.flagged = True

    model_rect_list = [rect_list[indx] for indx in model_indx]

    rect_list = [rect for e, rect in enumerate(rect_list) if e not in flagged_indx]

    return rect_list, flagged_rect_list, model_rect_list

def find_consecutive_in_range(array, lower_bound, upper_bound):
    count = 0
    start_indx = 0

    for i, val in enumerate(array):
        if lower_bound <= val <= upper_bound:
            if count == 0:
                start_indx = i
            count += 1
            if count == 50:
                return start_indx
        else:
            count = 0
    
    print("No point found within window increasing bounds")
    return find_consecutive_in_range(array, lower_bound * .5, upper_bound* 1.5)


def trim_boarder(img, direction):
    t_sum = np.sum(img, axis = direction)
    t_mode = find_mode(t_sum[t_sum != 0])
    lower_bound = t_mode - 10
    upper_bound = t_mode + 10
    min_index = find_consecutive_in_range(t_sum, lower_bound, upper_bound)
    max_index = find_consecutive_in_range(t_sum[::-1], lower_bound, upper_bound)
    max_index = t_sum.size - max_index
    
    if direction == 0:
        img[:, :(min_index + 1)] = 0
        img[:, max_index:] = 0
    else:
        img[:(min_index + 1), :] = 0
        img[max_index:, :] = 0
    
    return img


def compute_model(rect_list, img):
    width = rect_list[0].width
    height = rect_list[0].height

    model = np.zeros((height, width, 3))
    for rect in rect_list:
        sub_img = rect.create_sub_image(img)
        model = model + sub_img

    model = (model / len(rect_list)).astype(int)
    return model

def disp_rectangles_fig(rect_list, img):
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

def disp_rectangles_img(rect_list, img, name = False):
    output_img = np.copy(img)
    width = (rect_list[0].width / 2).astype(int)
    height = (rect_list[0].height / 2).astype(int)
    
    for rect in rect_list:
        top_left = (rect.center_x - width, rect.center_y - height)
        bottom_right = (rect.center_x + width, rect.center_y + height)

        if rect.flagged:
            output_img = cv.rectangle(output_img, top_left, bottom_right, (0, 255, 0), 10)
        else:
            output_img = cv.rectangle(output_img, top_left, bottom_right, (255, 0, 0), 10)

        if name:
            font = cv.FONT_HERSHEY_SIMPLEX
            scale = 1.5
            color = (255,255,255)
            position = (rect.center_x - width, rect.center_y)
            thickness = 5
            txt = str(rect.ID)
            output_img = cv.putText(output_img, txt, position, font, scale, color, thickness, cv.LINE_AA)
    
    output_img = Image.fromarray(output_img)

    return output_img
    
def correct_rect_range_row(rect_list):
    range_list = [rect.range for rect in rect_list]
    row_list = [rect.row for rect in rect_list]
    range_list = np.array(range_list)
    row_list = np.array(row_list)
    unique_range = np.unique(range_list)
    unique_row = np.unique(row_list)
    range_cnt = 1
    id_cnt = 1
    for range_indx in unique_range:
        row_cnt = 1
        for row_indx in unique_row:
            rect_indx = np.where((range_list == range_indx) & (row_list == row_indx))[0][0]
            rect = rect_list[rect_indx]
            rect.range = range_cnt
            rect.row = row_cnt
            rect.ID = id_cnt
            id_cnt += 1
            row_cnt += 1

        range_cnt += 1

    return rect_list

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


def find_center_line(img, poly_degree, direction):
    num_obj, labeled_img, _, _ = cv.connectedComponentsWithStats(img)
    skel = np.zeros_like(img)

    for e in range(1, num_obj):
        subset = np.column_stack(np.where(labeled_img == e))
        unique_values, counts = np.unique(subset[:, direction], return_counts=True)
        position = np.bincount(subset[:, direction], weights=subset[:, (1 - direction)])
        mean_values = (position[unique_values] / counts).astype(int)
        coefficients = np.polyfit(unique_values, mean_values, poly_degree)
        poly = np.poly1d(coefficients)
        x = np.arange(0, img.shape[direction], 1)
        y = poly(x).astype(int)

        if direction == 0:
            img_points = skel[x,y]
            if np.max(img_points) == 0:
                skel[x, y] = 1
        else:
            img_points = skel[y,x]
            if np.max(img_points) == 0:
                skel[y, x] = 1

    return skel
    
def impute_skel(skel, direction):
    if direction == 1:
        skel = skel.T
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    dialated_skel = cv.dilate(skel, kernel)

    num_obj, labeled_img, stats, centroids = cv.connectedComponentsWithStats(dialated_skel)
    # labeled_img[np.where(labeled_img == 59)] = 0
    # labeled_img[labeled_img != 0] = 1
    # labeled_img = labeled_img.astype(np.uint8)
    #  num_obj, labeled_img, stats, centroids = cv.connectedComponentsWithStats(labeled_img)

    center_distance = np.array([])
    for e in range(1, num_obj - 1):
        distance = abs(centroids[e, direction] - centroids[e + 1, direction]).astype(int)
        center_distance = np.append(center_distance, distance)

    median_dist = np.median(center_distance)
    avg_dist = (median_dist * 1.25).astype(int)

    obj_to_impute = np.where(center_distance > avg_dist)[0]
    impute_dist = (center_distance[obj_to_impute] / median_dist).astype(int)

    if obj_to_impute.size == 0:
        print("No objects to impute")
    else:
        indx = np.where(skel != 0)
        skel[indx] = labeled_img[indx]

        for e in range(obj_to_impute.size):
            top_indx = obj_to_impute[e] + 1
            bottom_indx = obj_to_impute[e] + 2
            top_side = np.column_stack(np.where(skel == top_indx))
            bottom_side = np.column_stack(np.where(skel == bottom_indx))
            top_side = top_side[np.argsort(top_side[:, 1])]
            bottom_side = bottom_side[np.argsort(bottom_side[:, 1])]
            step_size = ((bottom_side[:, 0] - top_side[:, 0]) / impute_dist[e]).astype(int)

            for k in range(impute_dist[e] - 1):
                new_obj = top_side
                new_obj[:, 0] = top_side[:, 0] + step_size * (k + 1)
                skel[new_obj[:, 0], new_obj[:, 1]] = 100

    skel[skel != 0] = 1
    skel = skel.astype(np.uint8)
    
    if direction == 1:
        skel = skel.T

    return skel

def remove_rectangles(rect_list, img, num_2_remove, direction):
    if direction != 1 and direction != 0:
        print("Invalid direction")
        exit()

    fft_scores = compute_fft_mat(rect_list, img)
    avg_fft_score = np.mean(fft_scores, axis = 0) 
    
    while num_2_remove > 0:
        if direction == 0:
            values = np.array([rect.row for rect in rect_list])
        else:
            values = np.array([rect.range for rect in rect_list])
        
        
        min_val_indx = np.where(values == np.min(values))[0]
        max_val_indx = np.where(values == np.max(values))[0]
        min_val_rect = [rect_list[indx] for indx in min_val_indx]
        max_val_rect = [rect_list[indx] for indx in max_val_indx]

        min_fft_mat = compute_fft_mat(min_val_rect, img)
        max_fft_mat = compute_fft_mat(max_val_rect, img)
        min_dist = compute_fft_distance(min_fft_mat, avg_fft_score)
        max_dist = compute_fft_distance(max_fft_mat, avg_fft_score)

        if min_dist > max_dist:
            rect_list = [rect for e, rect in enumerate(rect_list) if e not in min_val_indx]
        else:
            rect_list = [rect for e, rect in enumerate(rect_list) if e not in max_val_indx]
            
        num_2_remove -= 1

    return rect_list


def add_rectangles(rect_list, img, num_2_add, direction):
    if direction != 1 and direction != 0:
        print("Invalid direction")
        exit()

    fft_scores = compute_fft_mat(rect_list, img)
    avg_fft_score = np.mean(fft_scores, axis = 0) 

    while num_2_add > 0:
        if direction == 0:
            values = np.array([rect.row for rect in rect_list])
            delta = rect_list[0].width
        else:
            values = np.array([rect.range for rect in rect_list])
            delta = rect_list[0].height
     
        min_val = np.min(values)
        max_val = np.max(values)
        min_val_rect = np.where(values == min_val)[0]
        max_val_rect = np.where(values == max_val)[0]
        min_list = []
        max_list = []

        for e in range(len(min_val_rect)):
            tmp1_rect = deepcopy(rect_list[min_val_rect[e]])
            tmp2_rect = deepcopy(rect_list[max_val_rect[e]])

            if direction == 1:
                tmp1_rect.center_y = tmp1_rect.center_y - delta
                tmp2_rect.center_y = tmp2_rect.center_y + delta
                tmp1_rect.range = min_val - 1
                tmp2_rect.range = max_val + 1
            else:
                tmp1_rect.center_x = tmp1_rect.center_x - delta
                tmp2_rect.center_x = tmp2_rect.center_x + delta
                tmp1_rect.row = min_val - 1
                tmp2_rect.row = max_val + 1

            min_list.append(tmp1_rect)
            max_list.append(tmp2_rect)
        
        min_fft_mat = compute_fft_mat(min_list, img)
        max_fft_mat = compute_fft_mat(max_list, img)
        min_fft_dist = compute_fft_distance(min_fft_mat, avg_fft_score)
        max_fft_dist = compute_fft_distance(max_fft_mat, avg_fft_score)

        if min_fft_dist < max_fft_dist:
            rect_list = rect_list + min_list
        else:
            rect_list = rect_list + max_list

        num_2_add -= 1

    return rect_list


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

def create_unit_square(width, height):
    # Creating the unit square
    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    X, Y = np.meshgrid(x, y)
    unit_sqr = np.column_stack((X.ravel(), Y.ravel(), np.ones_like(X.ravel())))

    return unit_sqr 


def create_affine_frame(center_x, center_y, theta, width, height):
    if width != 0:
        width = np.array(width / 2).astype(int)
    if height != 0:
        height = np.array(height / 2).astype(int)
    #theta = np.radians(theta)

    # Translation Matrix
    t_mat = np.zeros((3, 3))
    t_mat[0, 0], t_mat[1, 1], t_mat[2, 2] = 1, 1, 1
    t_mat[0, 2], t_mat[1, 2] = center_x, center_y

    # Scaler Matrix
    s_mat = np.zeros((3, 3))
    s_mat[0, 0], s_mat[1, 1], s_mat[2, 2] = width, height, 1

    # Rotation Matrix
    r_1 = [np.cos(theta), np.sin(theta), 0]
    r_2 = [-np.sin(theta), np.cos(theta), 0]
    r_3 = [0, 0, 1]
    r_mat = np.column_stack((r_1, r_2, r_3))

    affine_mat = t_mat @ r_mat @ s_mat
    return affine_mat

def compute_points(center_x, center_y, theta, width, height, unit_sqr, img_shape):
    affine_mat = create_affine_frame(center_x, center_y, theta, width, height)
    rotated_points = np.dot(affine_mat, unit_sqr.T).T
    rotated_points = rotated_points[:,:2].astype(int)

   
    # Checking to make sure points are within the image
    img_height, img_width = img_shape[:2]
    valid_y = (rotated_points[:, 1] >= 0) & (rotated_points[:, 1] < img_height)
    valid_x = (rotated_points[:, 0] >= 0) & (rotated_points[:, 0] < img_width)
    invalid_points = (~(valid_x & valid_y))
    rotated_points[invalid_points, :] = [0,0]

    return rotated_points


def extract_rectangle(center_x, center_y, theta, width, height, unit_sqr, img):
    points = compute_points(center_x, center_y, theta, width, height, unit_sqr, img.shape)

    if len(img.shape) > 2:
        extracted_img = img[points[:, 1], points[:, 0], :]
        extracted_img = np.reshape(extracted_img, (height, width, img.shape[2]))
    else:
        extracted_img = img[points[:, 1], points[:, 0]]
        extracted_img = np.reshape(extracted_img, (height, width))
        
    return extracted_img
             


def single_gaussian(x, mu, sigma):
    return norm.pdf(x, mu, sigma)

def bimodal_gaussian(x, mu1, sigma1, mu2, sigma2, weight):
    return weight * norm.pdf(x, mu1, sigma1) + (1 - weight) * norm.pdf(x, mu2, sigma2)
            
def trimodal_gaussian(x, mu1, sigma1, mu2, sigma2, mu3, sigma3, w1, w2):
    return (w1 * norm.pdf(x, mu1, sigma1) + 
            w2 * norm.pdf(x, mu2, sigma2) + 
            (1 - w1 - w2) * norm.pdf(x, mu3, sigma3))

# Calc the AIC function
def calc_aic(y, yhat, k):
    resid = y - yhat
    sse = np.sum(resid**2)
    n = len(y)
    out = 2*k + n * np.log(sse/n)
    return out

# Calculating the probability of each pixel belonging to the distributions
def calculate_prob(x, params):
    mu1, sigma1, mu2, sigma2, weight = params
    pdf1 = weight * norm.pdf(x, mu1, sigma1)
    pdf2 = (1 - weight) * norm.pdf(x, mu2, sigma2)
    total_prob = pdf1 + pdf2
    prob1 = pdf1 / total_prob
    prob2 = pdf2 / total_prob
    return prob1, prob2