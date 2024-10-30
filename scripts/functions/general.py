import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd

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
    
def find_consecutive_in_range(array, lower_bound, upper_bound, num_consecutive):
    count = 0
    start_indx = 0

    for i, val in enumerate(array):
        if lower_bound <= val <= upper_bound:
            if count == 0:
                start_indx = i
            count += 1
            if count == num_consecutive:
                return start_indx
        else:
            count = 0
    
    print("No point found within window increasing bounds")
    return find_consecutive_in_range(array, lower_bound * .8, upper_bound* 1.2, num_consecutive)

def create_shapefile(rect_list, original_transform, original_crs, inverse_rotation_matrix, file_name):
    poly_data = []
    for rect in rect_list:
        points = rect.compute_corner_points()
        points = np.column_stack((points, np.ones(points.shape[0])))

        if inverse_rotation_matrix is not None:
            points = np.dot(inverse_rotation_matrix, points.T).T
            
        points = original_transform * points.T
        points = tuple(zip(points[0], points[1]))
        temp_poly = Polygon(points)
        poly_data.append({'geometry': temp_poly, 'label': rect.ID})
    
    gdf = gpd.GeoDataFrame(poly_data, crs=original_crs)
    gdf.to_file(file_name, driver="GPKG")

    return

def geometric_median(points, weights=None, tol = 1e-2):
    points = np.asarray(points)
    if weights is None:
        weights = np.ones(len(points))
    else:
        weights = np.asarray(weights)
    
    guess = np.mean(points, axis = 0)

    while True:
        distances = np.linalg.norm(points - guess, axis=1)
        nonzero = (distances != 0)
        
        if not np.any(nonzero):
            return guess
        
        w = weights[nonzero] / distances[nonzero]
        new_guess = np.sum(points[nonzero] * w[:, None], axis=0) / np.sum(w)
        
        if np.linalg.norm(new_guess - guess) < tol:
            return new_guess
        
        guess = new_guess