import os, multiprocessing
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from functions import find_mode, find_center_line
from PIL import Image

class wavepad:
    def __init__(self, row_wavepad_binary, range_wavepad_binary, QC_output, ncore):
        self.row_wavepad_binary = row_wavepad_binary
        self.range_wavepad_binary = range_wavepad_binary
        self.output_path = QC_output
        self.ncore = ncore

    def find_ranges(self, poly_degree):
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

    def find_columns(self, poly_degree):
        # Closing the Image
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20,20))
        tmp = cv.morphologyEx(self.row_wavepad_binary, cv.MORPH_CLOSE, kernel)
        tmp = cv.GaussianBlur(tmp, (5,5),2)

        # Finding Image object stats
        _, obj_filtered_wavepad, img_stats, _ = cv.connectedComponentsWithStats(tmp)
        object_areas = img_stats[:,4]

        # Finding the correct size objects
        normalized_areas = np.zeros((object_areas.size,object_areas.size))
        for e in range(object_areas.size):
            for k in range(object_areas.size):
                normalized_areas[e,k] = round(object_areas[e] / object_areas[k])

        sum_count = np.sum(normalized_areas == 1, axis = 1)
        mode_count = find_mode(sum_count)
        find_indx = np.where(sum_count == mode_count)[0]
        mu = np.mean(object_areas[find_indx])

        correct_indx = np.zeros_like(object_areas)
        for k in range(object_areas.size):
            rel_object_size = round(object_areas[k] / mu)
            if rel_object_size > .9 and rel_object_size < 1.25:
                correct_indx[k] = 1
            else:
                correct_indx[k] = 0

        # Filtering out incorrect size objects
        mask = np.isin(obj_filtered_wavepad, np.where(correct_indx == 1))
        obj_filtered_wavepad = np.where(mask, 1,0).astype(np.uint8)

        # Saving the output
        name = 'Row_Object_Filtered_Wavepad.jpg'
        Image.fromarray(obj_filtered_wavepad * 255).save(os.path.join(self.output_path,name))

        # Finding the objects
        num_obj, obj_labeled_img, img_stats, _ = cv.connectedComponentsWithStats(obj_filtered_wavepad)

        # Looping through all objects to draw center line
        print(f"Using {self.ncore} cores to find center lines")
        with multiprocessing.Pool(processes=self.ncore) as pool:
            centerlines = pool.map(
                find_center_line, [(obj_labeled_img, indx, poly_degree, obj_labeled_img.shape[0]) for indx in range(1,num_obj)]
            )
      
        # Create Skeleton
        skel = np.zeros_like(obj_filtered_wavepad)
        center_points = np.array(centerlines)
        center_x = center_points[:,:,1].flatten()
        center_y = center_points[:,:,0].flatten()
        skel[center_x, center_y] = 1
        self.real_col_skel = skel

        # Return Output
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
        dialated_skel = cv.dilate(skel.astype(np.uint8), kernel)
        return dialated_skel

    def imput_col_skel(self):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
        tmp = cv.dilate(self.real_col_skel, kernel)
        num_col, labeled_skel, stats, centroids = cv.connectedComponentsWithStats(np.transpose(tmp))
        center_distance = np.array([])
        for e in range(1, centroids.shape[0] - 1):
            distance = abs(centroids[e,1] - centroids[e + 1,1])
            center_distance = np.append(center_distance, distance)

        center_distance = np.round(center_distance).astype(int)
        avg_distance = find_mode(center_distance)
        self.avg_col_spacing = avg_distance * 1.25
        #plt.scatter(np.arange(center_distance.size), center_distance)

        col_to_imput = np.where(center_distance > avg_distance * 1.25)[0]
        num_cols_to_imput = np.round(center_distance[center_distance > avg_distance * 1.25] / avg_distance).astype(int)

        #Return if no columns to impute
        if col_to_imput.size == 0:
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
            dialated_skel = cv.dilate(self.real_col_skel.astype(np.uint8), kernel)
            return dialated_skel, self.real_col_skel
        else:
            test = np.copy(self.real_col_skel.T)
            indx = np.argwhere(self.real_col_skel.T != 0)
            test[indx[:,0],indx[:,1]] = labeled_skel[indx[:,0],indx[:,1]]
            labeled_skel = test
            fake_col_skel = []
            for e in range(col_to_imput.size):
                top_side = np.column_stack(np.where(labeled_skel == (col_to_imput[e] + 1)))
                bottom_side = np.column_stack(np.where(labeled_skel == (col_to_imput[e] + 2)))
                top_side = top_side[np.argsort(top_side[:,1])]
                bottom_side = bottom_side[np.argsort(bottom_side[:,1])]

                # Avoiding error with vectors lengths
                if top_side.shape[0] != bottom_side.shape[0]:
                    if top_side.shape[0] > bottom_side.shape[0]:
                        max_size = bottom_side.shape[0]
                    else:
                        max_size = top_side.shape[0]
                else:
                    max_size = top_side.shape[0]

                top_y_cord = top_side[:max_size,0]
                bottom_y_cord = bottom_side[:max_size,0]
                step_size = np.round((bottom_y_cord - top_y_cord) / num_cols_to_imput[e]).astype(int)

                for k in range(num_cols_to_imput[e] - 1):
                    output = np.zeros((max_size,2)).astype(int)
                    output[:,0] = top_side[:max_size,1]
                    output[:,1] = top_side[:max_size,0] + step_size * (k + 1)
                    fake_col_skel.append(output)

            fake_col_skel = np.concatenate(fake_col_skel, axis = 0)
            col_skel = self.real_col_skel
            col_skel[fake_col_skel[:,0],fake_col_skel[:,1]] = 1

            #Creating output to return
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
            dialated_skel = cv.dilate(self.col_skel, kernel)
            return dialated_skel, col_skel


