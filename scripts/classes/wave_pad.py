import os
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from PIL import Image

from functions.wavepad import filter_wavepad, trim_boarder, find_center_line, impute_skel
from functions.image_processing import find_correct_sized_obj
from functions.rectangle import correct_rect_range_row, build_rectangles, find_next_rect, find_edge_rect
from classes.rectangles import rect_list


class wavepad:
    def __init__(self, range_wavepad, row_wavepad, params, img):
        self.params = params
        self.range_wavepad = range_wavepad
        self.row_wavepad = row_wavepad
        self.img = img
        self.phase_one() # Filtering the wavepads
        self.phase_two() # Finding the center lines and initial rectangles
        self.phase_three() # Finding the all rectangles
        #self.phase_four() # Making sure we found the correct ranges/rows
        self.phase_five() # add labels and export

    def phase_one(self):
        #Pull the params
        filter_method = self.params["wavepad_filter_method"]

        # Filtering the wavepads
        self.filtered_range_wp = filter_wavepad(self.range_wavepad, filter_method)
        self.filtered_row_wp = filter_wavepad(self.row_wavepad, filter_method)

        # triming the boarder
        self.filtered_range_wp = trim_boarder(self.filtered_range_wp, 0)
        self.filtered_row_wp = trim_boarder(self.filtered_row_wp, 1)

        # Finding the correct sized objects
        self.filtered_range_wp = find_correct_sized_obj(self.filtered_range_wp)
        self.filtered_row_wp = find_correct_sized_obj(self.filtered_row_wp)

        print("Finished Filtering Wavepads")

    def phase_two(self):
        # Pull the params
        poly_deg_range = self.params["poly_deg_range"]
        poly_deg_row = self.params["poly_deg_row"]
        range_cnt = self.params["nranges"]
        row_cnt = self.params["nrows"]

        # Finding the center lines
        range_center_lines = find_center_line(self.filtered_range_wp, poly_deg_range, 1) # 1 for range
        row_center_lines = find_center_line(self.filtered_row_wp, poly_deg_row, 0) # 0 for row

        # imputing missing rows/ ranges
        impute_range_skel = impute_skel(range_center_lines, 0)
        impute_row_skel = impute_skel(row_center_lines, 1)

        # Finding the initial rectangles
        initial_rect_list, initial_range_cnt, initial_row_cnt = build_rectangles(impute_range_skel, impute_row_skel)
        
        # Creating the rectangle list object
        self.initial_rect_list = rect_list(initial_rect_list, self.img, build_rectangles = True)
        self.initial_rect_list.build_model() # Building the model

        # Computing the number of ranges and rows to find
        self.ranges_2_find = range_cnt - initial_range_cnt
        self.rows_2_find = row_cnt - initial_row_cnt

        print("Finished Finding Center Lines and Initial Rectangles")

    def phase_three(self):
        if self.rows_2_find == 0 and self.ranges_2_find == 0:
            print("All plots found from FFT")

        # Finding the correct number of ranges
        if self.ranges_2_find < 0:
            print(f"Removing {abs(self.ranges_2_find)} extra range(s) from FFT")
            self.remove_rectangles("range", abs(self.ranges_2_find))
        else:
            print(f"Finding {self.ranges_2_find} missing range(s) from FFT")
            self.add_rectangles("range", self.ranges_2_find)
        
        # Finding the correct number of rows
        if self.rows_2_find < 0:
            print(f"Removing {abs(self.rows_2_find)} extra row(s) from FFT")
            self.remove_rectangles("row", abs(self.rows_2_find))
        else:
            print(f"Finding {self.rows_2_find} missing row(s) from FFT")
            self.add_rectangles("row", self.rows_2_find)

        print("Finished Finding All Rectangles")
    
    def add_rectangles(self, direction, num_2_add):
        for _ in range(num_2_add):
            # Find the next rectangles
            min_list, max_list = find_next_rect(self.initial_rect_list, direction)

            # Create the rectangle list object
            min_list = rect_list(min_list, self.img, build_rectangles = False)
            max_list = rect_list(max_list, self.img, build_rectangles = False)

            # Adding the model from the initial rect list
            min_list.model = self.initial_rect_list.model
            max_list.model = self.initial_rect_list.model

            # Computing the scores
            min_score = min_list.compute_score()
            max_score = max_list.compute_score()

            # Adding the rectangles with min score
            if min_score >= max_score:
                self.initial_rect_list.add_rectangles(max_list.rect_list)
            else:
                self.initial_rect_list.add_rectangles(min_list.rect_list)

        print(f"Added {num_2_add} rectangles in the {direction} direction")

    def remove_rectangles(self, direction, num_2_remove):
        for _ in range(num_2_remove):
            # Find the current min, max rectangles
            min_list, max_list = find_edge_rect(self.initial_rect_list, direction)

            # Creating the rect list object
            min_val_rect = rect_list(min_list, self.img, build_rectangles = False)
            max_val_rect = rect_list(max_list, self.img, build_rectangles = False)
            min_val_rect.model = self.initial_rect_list.model
            max_val_rect.model = self.initial_rect_list.model

            # Computing the scores
            min_score = min_val_rect.compute_score()
            max_score = max_val_rect.compute_score()

            if min_score <= max_score:
                self.initial_rect_list.remove_rectangles(min_val_rect.rect_list)
            else:
                self.initial_rect_list.remove_rectangles(max_val_rect.rect_list)
        
        print(f"Removed {num_2_remove} rectangles in the {direction} direction")

    def phase_four(self):
        # Checking in the row direction
        flag = True
        while flag:
            current_edge_min, current_edge_max = find_edge_rect(self.initial_rect_list, "row")
            next_row_min, next_row_max = find_next_rect(self.initial_rect_list, "row")

            rect_list_to_test = [current_edge_min, current_edge_max, next_row_min, next_row_max]
            scores = []

            for tmp_list in rect_list_to_test:
                tmp_list = rect_list(tmp_list, self.img, build_rectangles = False)
                tmp_list.model = self.initial_rect_list.model
                
                param_dict = {}
                param_dict['method'] = 'PSO'
                param_dict['swarm_size'] = 20
                param_dict['maxiter'] = 100
                param_dict['x_radi'] = 20
                param_dict['y_radi'] = 20
                param_dict['theta_radi'] = 5

                tmp_list.optimize_rectangles(param_dict)

                tmp_list_score = tmp_list.compute_score()
                scores.append(tmp_list_score)
                
            if scores[2] < scores[1]:
                self.initial_rect_list.add_rectangles(next_row_min)
                self.initial_rect_list.remove_rectangles(current_edge_max)
            elif scores[3] < scores[0]:
                self.initial_rect_list.add_rectangles(next_row_max)
                self.initial_rect_list.remove_rectangles(current_edge_min)
            else:
                flag = False

        print("Finished Double Checking Rows")


    def phase_five(self):
        self.final_rect_list = correct_rect_range_row(self.initial_rect_list)
    
    
