import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from functions import find_center_line, find_correct_sized_obj, trim_boarder, impute_skel, build_rectangles, add_rectangles, remove_rectangles, correct_rect_range_row
from PIL import Image
from operator import itemgetter
from rectangles import rectangle


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

    def find_ranges(self):
        # Closing the Image
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
        range_wavepad_binary = cv.morphologyEx(self.range_wavepad_binary, cv.MORPH_CLOSE, kernel)
        range_wavepad_binary = cv.GaussianBlur(range_wavepad_binary, (5, 5), 2)

        # Trimming broaders
        range_wavepad_binary = trim_boarder(range_wavepad_binary, 0)

        # Finding the correct sized objects
        obj_filtered_range_wavepad = find_correct_sized_obj(range_wavepad_binary)
        
        # Saving QC Output
        if self.params["QC_depth"] == "max":
            name = 'Range_Object_Filtered_Wavepad.jpg'
            Image.fromarray((obj_filtered_range_wavepad * 255).astype(int)).save(os.path.join(self.output_path,name))

       # Finding the center lines
        center_lines_range_wavepad = find_center_line(obj_filtered_range_wavepad, self.params["poly_deg_range"], 1)

        # imputing missing columns
        final_range_skel = impute_skel(center_lines_range_wavepad, 0)

        # Return Output
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
        dialated_skel = cv.dilate(final_range_skel, kernel)
        self.final_range_skel = final_range_skel

        return dialated_skel


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
        self.final_row_skel = final_row_skel

        return dialated_skel
    
    def find_plots(self, img):
        fft_rect, range_cnt, row_cnt = build_rectangles(self.final_range_skel, self.final_row_skel)

        mean_width = np.mean(np.array(list(map(itemgetter(2), fft_rect)))).astype(int)
        mean_height = np.mean(np.array(list(map(itemgetter(3), fft_rect)))).astype(int)

        for e, rect in enumerate(fft_rect):
            rect = rectangle(rect)
            rect.width = mean_width
            rect.height = mean_height
            fft_rect[e] = rect

        rows_2_find = self.params["nrows"] - row_cnt
        ranges_2_find = self.params["nranges"] - range_cnt

        if rows_2_find == 0 and ranges_2_find == 0:
            print("All plots found from FFT")

        if ranges_2_find < 0:
            print(f"Removing {abs(ranges_2_find)} extra range(s) from FFT")
            fft_rect =  remove_rectangles(fft_rect, img, abs(ranges_2_find), 1)
        else:
            print(f"Finding {ranges_2_find} missing range(s) from FFT")
            fft_rect = add_rectangles(fft_rect, img, ranges_2_find, 1)
            
        if rows_2_find < 0:
            print(f"Removing {abs(rows_2_find)} extra row(s) from FFT")
            fft_rect = remove_rectangles(fft_rect, img,  abs(rows_2_find), 0)
        else:
            print(f"Finding {rows_2_find} missing row(s) from FFT")
            fft_rect = add_rectangles(fft_rect, img, rows_2_find, 0)

        fft_rect = correct_rect_range_row(fft_rect)

        return fft_rect

        

