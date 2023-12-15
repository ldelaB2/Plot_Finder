import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from functions import find_mode, break_up_skel
from PIL import Image
from skimage.morphology import skeletonize
#from plantcv import plantcv as pcv
import itertools
import multiprocessing

class wavepad:
    def __init__(self, row_wavepad_binary, range_wavepad_binary, QC_output):
        self.row_wavepad_binary = row_wavepad_binary
        self.range_wavepad_binary = range_wavepad_binary
        self.output_path = QC_output


    def compute_rectangles(self):
        print("T")
        num_ranges, range_labled, img_stats, _ = cv.connectedComponentsWithStats(self.real_range_skel)


        range_bool = self.real_range_skel.astype(bool)
        row_bool = self.col_skel.astype(bool)
        cp = np.argwhere(range_bool & row_bool)
        cp_y_sorted = cp[np.argsort(cp[:,0])]
        cp_x_sorted = cp[np.argsort(cp[:, 1])]

    def find_ranges(self, ncore):
        # This totally is not going to work
        # Closing the Image
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
        tmp = cv.morphologyEx(self.range_wavepad_binary, cv.MORPH_CLOSE, kernel)

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
            if rel_object_size > .9 and rel_object_size < 2:
                correct_indx[k] = 1
            else:
                correct_indx[k] = 0

        # Filtering out incorrect size objects
        mask = np.isin(obj_filtered_wavepad, np.where(correct_indx == 1))
        obj_filtered_wavepad = np.where(mask, 1, 0).astype(np.uint8)

        # Skeletonizing
        skel = skeletonize(obj_filtered_wavepad.astype(bool))

        # Breaking up the skeleton
        binary_matrices = list(itertools.product([0,1],repeat = 9))
        kernels = [truple for truple in binary_matrices if truple.count(1) > 3]

        print(f"Using {ncore} cores to prune range wavepad")
        with multiprocessing.Pool(processes = ncore) as pool:
            results = pool.map(
                break_up_skel, [(kernel, skel, [3,3]) for kernel in kernels]
            )

        unique_points = [tuple(point) for sublist in results if sublist is not None for point in sublist]
        unique_points = np.array(list(set(unique_points)))
        broken_skel = skel.astype(np.uint8)
        broken_skel[unique_points[:,0],unique_points[:,1]] = 0

        # Filtering out objects under 300 pixels
        num_objects, broken_skel_labeled, img_stats, _ = cv.connectedComponentsWithStats(broken_skel)
        object_areas = img_stats[:, 4]
        area_thresh = 300
        obj_to_remove = np.where(object_areas < area_thresh)
        broken_skel_labeled[np.isin(broken_skel_labeled,obj_to_remove)] = 0
        broken_skel_labeled[broken_skel_labeled > 0] = 1

        # Triming broaders
        temp = np.sum(broken_skel_labeled, 0)
        temp_mode = find_mode(temp[temp != 0])
        temp_index = np.where(temp == temp_mode)
        min_index = np.min(temp_index)
        max_index = np.max(temp_index)
        broken_skel_labeled[:, :(min_index + 1)] = 0
        broken_skel_labeled[:, max_index:] = 0

        #Dialating to fill holes
        num_iterations = 20
        tmp = broken_skel_labeled.astype(np.uint8)
        kernel1 = np.ones((10,50))
        for e in range(num_iterations):
            tmp = cv.dilate(tmp, kernel1)

        tmp[:,:(min_index + 1)] = 0
        tmp[:, max_index:] = 0
        skel = skeletonize(tmp.astype(bool))
        self.real_range_skel = skel

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 50))
        dialated_skel = cv.dilate(skel.astype(np.uint8), kernel)
        return dialated_skel


    def find_columns(self):
        # Closing the Image
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20,20))
        tmp = cv.morphologyEx(self.row_wavepad_binary, cv.MORPH_CLOSE, kernel)

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
            if rel_object_size > .9 and rel_object_size < 2:
                correct_indx[k] = 1
            else:
                correct_indx[k] = 0

        # Filtering out incorrect size objects
        mask = np.isin(obj_filtered_wavepad, np.where(correct_indx == 1))
        obj_filtered_wavepad = np.where(mask, 1,0).astype(np.uint8)

        # Saving the output
        name = 'Row_Object_Filtered_Wavepad.jpg'
        Image.fromarray(obj_filtered_wavepad * 255).save(os.path.join(self.output_path,name))

        # Skeletonize and prune
        skel = skeletonize(obj_filtered_wavepad.astype(bool))
        skel = pcv.morphology.prune(skel.astype(np.uint8), size = 200)[0]

        # Trimming boarder off
        temp = np.sum(skel, 1)
        temp_mode = find_mode(temp[temp != 0])
        temp_index = np.where(temp == temp_mode)
        min_index = np.min(temp_index)
        max_index = np.max(temp_index)
        skel[0:(min_index + 1),:] = 0
        skel[max_index:skel.shape[0],:] = 0
        self.real_col_skel = skel

        # Return Output
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
        dialated_skel = cv.dilate(skel.astype(np.uint8), kernel)
        return dialated_skel

    def imput_col_skel(self):
        num_col, labeled_skel, stats, centroids = cv.connectedComponentsWithStats(np.transpose(self.real_col_skel))
        center_distance = np.array([])
        for e in range(1, centroids.shape[0] - 1):
            distance = abs(centroids[e,1] - centroids[e + 1,1])
            center_distance = np.append(center_distance, distance)

        center_distance = np.round(center_distance).astype(int)
        avg_distance = find_mode(center_distance)
        #plt.scatter(np.arange(center_distance.size), center_distance)

        col_to_imput = np.where(center_distance > avg_distance * 1.25)[0]
        num_cols_to_imput = np.round(center_distance[center_distance > avg_distance * 1.25] / avg_distance).astype(int)

        #Return if no columns to impute
        if col_to_imput.size == 0:
            return
        else:
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
            self.col_skel = self.real_col_skel
            self.col_skel[fake_col_skel[:,0],fake_col_skel[:,1]] = 1

            #Creating output to return
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
            dialated_skel = cv.dilate(self.col_skel, kernel)
            return dialated_skel


