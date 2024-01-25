import os, multiprocessing
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from functions import find_mode, find_center_line, find_correct_sized_obj, trim_boarder, impute_skel
from PIL import Image

class wavepad:
    def __init__(self, row_wavepad_binary, range_wavepad_binary, QC_output, params):
        """
        Initializes the wavepad object.

        Parameters:
            row_wavepad_binary (ndarray): The row wavepad binary image.
            range_wavepad_binary (ndarray): The range wavepad binary image.
            QC_output (str): The path to the output directory for quality control.
            ncore (int): The number of cores to use for parallel processing.

        The function initializes the wavepad object with the provided row and range wavepad binary images, output path, and number of cores.
        """
        self.row_wavepad_binary = row_wavepad_binary
        self.range_wavepad_binary = range_wavepad_binary
        self.output_path = QC_output
        self.params = params

    def find_ranges(self, poly_degree):
        """
        This method finds the ranges in the wavepad image.

        Parameters:
            poly_degree (int): The degree of the polynomial to fit to the mean 'y' values for each 'x' in the skeleton image.

        Returns:
            tuple: A tuple containing the dilated skeleton image and the original skeleton image.

        The method first closes and blurs the range wavepad binary image.
        It then trims the borders of the image and finds the connected components in the image.
        It calculates the normalized areas of the connected components and finds the components that are the correct size.
        It then filters out the incorrect size components and saves the filtered image.
        It finds the connected components in the filtered image and draws a center line through each component by fitting a polynomial to the mean 'y' values for each 'x'.
        Finally, it dilates the skeleton image and returns the dilated image and the original skeleton image.
        """

        # Closing the Image
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
        tmp = cv.morphologyEx(self.range_wavepad_binary, cv.MORPH_CLOSE, kernel)
        tmp = cv.GaussianBlur(tmp, (5, 5), 2)

        # Trimming broaders
        t_sum = np.sum(tmp, 0)
        t_mode = find_mode(t_sum[t_sum != 0])
        t_index = np.where(t_sum == t_mode)
        min_index = np.min(t_index)
        max_index = np.max(t_index)
        tmp[:, :(min_index + 1)] = 0
        tmp[:, max_index:] = 0

        # Finding Image object stats
        _, obj_filtered_wavepad, img_stats, _ = cv.connectedComponentsWithStats(tmp)
        object_areas = img_stats[:, 4]

        # Finding the correct size objects
        normalized_areas = np.zeros((object_areas.size - 1, object_areas.size - 1))
        for e in range(1,object_areas.size):
            for k in range(1,object_areas.size):
                normalized_areas[e - 1, k - 1] = round(object_areas[e] / object_areas[k])

        sum_count = np.sum(normalized_areas == 1, axis=1)
        mode_count = find_mode(sum_count)
        find_indx = np.where(sum_count == mode_count)[0]
        mu = np.mean(object_areas[find_indx + 1])

        correct_indx = np.zeros_like(object_areas)
        for k in range(object_areas.size):
            rel_object_size = round(object_areas[k] / mu)
            if rel_object_size > .75 and rel_object_size < 2:
                correct_indx[k] = 1
            else:
                correct_indx[k] = 0

        # Filtering out incorrect size objects
        mask = np.isin(obj_filtered_wavepad, np.where(correct_indx == 1))
        obj_filtered_wavepad = np.where(mask, 1, 0).astype(np.uint8)

        # Saving the output
        name = 'Range_Object_Filtered_Wavepad.jpg'
        Image.fromarray(obj_filtered_wavepad * 255).save(os.path.join(self.output_path, name))

        # Finding the objects
        num_obj, obj_labeled_img, img_stats, _ = cv.connectedComponentsWithStats(obj_filtered_wavepad)

        # Looping through all objects to draw center line
        skel = np.zeros_like(obj_filtered_wavepad)

        for e in range(1, num_obj):
            subset_x = np.column_stack(np.where(obj_labeled_img == e))
            # Calculate mean 'y' for each 'x'
            unique_vales, counts = np.unique(subset_x[:, 1], return_counts=True)
            y_position = np.bincount(subset_x[:, 1], weights=subset_x[:, 0])
            mean_y_values = (y_position[unique_vales] / counts).astype(int)
            # Fit a polynomial to these points
            coefficients = np.polyfit(unique_vales, mean_y_values, poly_degree)
            poly = np.poly1d(coefficients)
            # Get the x and y coordinates
            x = np.arange(0, skel.shape[1], 1)
            y = poly(x).astype(int)
            # Set the corresponding points in skel to 1
            skel[y, x] = 1
          
        # Returning output
        self.real_range_skel = skel
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20,20))
        dialated_skel = cv.dilate(skel.astype(np.uint8), kernel)
        return dialated_skel, skel

    def find_rows(self):
        # Closing the Image
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20,20))
        row_wavepad_binary = cv.morphologyEx(self.row_wavepad_binary, cv.MORPH_CLOSE, kernel)
        row_wavepad_binary = cv.GaussianBlur(row_wavepad_binary, (5,5),2)
        
        # Trimming broaders
        row_wavepad_binary = trim_boarder(row_wavepad_binary, 1)

        # Finding the correct sized objects
        obj_filtered_row_wavepad = find_correct_sized_obj(row_wavepad_binary)
        
        # Saving QC Output
        if self.params["QC_depth"] == "max":
            name = 'Row_Object_Filtered_Wavepad.jpg'
            Image.fromarray((obj_filtered_row_wavepad * 255).astype(int)).save(os.path.join(self.output_path,name))

        # Finding the center lines
        center_lines_row_wavepad = find_center_line(obj_filtered_row_wavepad, self.params["poly_deg_row"], 0)
        
        # imputing missing columns
        final_row_skel = impute_skel(center_lines_row_wavepad, 1)

        # Return Output
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
        dialated_skel = cv.dilate(final_row_skel, kernel)
        return dialated_skel