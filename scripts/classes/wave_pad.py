import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

from functions.wavepad import filter_wavepad, trim_boarder, find_center_line, impute_skel
from functions.image_processing import find_correct_sized_obj
from functions.rectangle import set_range_row, check_within_img, compute_score_list, compare_next_to_current, set_id, build_rectangles, find_next_rect, remove_rectangles
from functions.optimization import build_rect_list, compute_model, optimize_list
from functions.display import disp_rectangles
import numpy as np


class wavepad:
    def __init__(self, range_wavepad, row_wavepad, params, img):
        self.params = params
        self.range_wavepad = range_wavepad
        self.row_wavepad = row_wavepad
        self.img = img

        self.phase_one() # Filtering the wavepads
        self.phase_two() # Finding the center lines and initial rectangles
        self.phase_three() # Finding the all rectangles
        self.phase_four() # Making sure we found the correct ranges/rows
        self.phase_five() # add labels

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
        
        # Building the rect list and model
        self.initial_rect_list = build_rect_list(initial_rect_list, self.img)
        self.initial_model = compute_model(self.initial_rect_list)
        
        # Optimize initial rectangles
        self.sparse_optimize(self.initial_rect_list, self.initial_model, recompute_model = True, txt = "Initial Sparse Optimization")

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
        self.final_rect_list = self.initial_rect_list

    def phase_four(self):
        # Checking to make sure we found the correct ranges and rows
        row_flag = True
        range_flag = True
        range_cnt = 0
        row_cnt = 0

        while row_flag or range_flag:
            if row_flag:
                direction = "row"
                update_flag, next_best_list, current_best_list = compare_next_to_current(self.final_rect_list, self.initial_model, direction)
                if update_flag:
                    # Remove the current best list and add the next best list
                    self.final_rect_list = remove_rectangles(self.final_rect_list, current_best_list)
                    for rect in next_best_list:
                        self.final_rect_list.append(rect)

                    row_cnt += 1
                else:
                    row_flag = False

            if range_flag:
                direction = "range"
                update_flag, next_best_list, current_best_list = compare_next_to_current(self.final_rect_list, self.initial_model, direction)
                if update_flag:
                    # Remove the current best list and add the next best list
                    self.final_rect_list = remove_rectangles(self.final_rect_list, current_best_list)
                    for rect in next_best_list:
                        self.final_rect_list.append(rect)

                    range_cnt += 1
                else:
                    range_flag = False
        
        print(f"Shifted {row_cnt} row(s) and {range_cnt} range(s) on double check")

    def phase_five(self):
        # Pulling the params
        range_cnt = self.params["nranges"]
        row_cnt = self.params["nrows"]
        label_start = self.params["label_start"]
        label_flow = self.params["label_flow"]

        # Setting the range and row
        set_range_row(self.final_rect_list, range_cnt, row_cnt)
        # Setting the id
        set_id(self.final_rect_list, start = label_start, flow = label_flow)
        
        print("Finished Adding Labels")


    def sparse_optimize(self, rect_list, model, txt = "Sparse Optimization", recompute_model = False):
        opt_param_dict = {}
        opt_param_dict['method'] = 'feature'
        opt_param_dict['x_radi'] = 10
        opt_param_dict['y_radi'] = 10
        opt_param_dict['feature_num_features'] = 100
        opt_param_dict['feature_num_points'] = 10
        opt_param_dict['feature_quality'] = .1
        opt_param_dict['feature_min_dist'] = 7
        opt_param_dict['feature_block_size'] = 7


        # Optimize the rectangles
        optimize_list(rect_list, model, opt_param_dict, txt)
        
        if recompute_model:
            self.initial_model = compute_model(rect_list)
            print("Recomputed Model")

    def add_rectangles(self, direction, num_2_add):
        for cnt in range(num_2_add):
            # Find the next rectangles
            min_list, max_list = find_next_rect(self.initial_rect_list, direction, edge = False)

            # Check to make sure the rectangles are within the image
            min_mult, max_mult = check_within_img(min_list, max_list, self.img.shape)

            if min_mult != np.inf:
                # Optimize the rectangles
                self.sparse_optimize(min_list, self.initial_model, txt = f"Sparse Optimization Add Min {direction} {cnt + 1}/{num_2_add}")
                # Compute the score
                min_score = compute_score_list(min_list, self.initial_model)
                # Multiply the score by the multiplier
                min_score = min_score * min_mult
            else:
                min_score = np.inf
                print("Min Rectangles are out of bounds")

            if max_mult != np.inf:
                # Optimize the rectangles
                self.sparse_optimize(max_list, self.initial_model, txt = f"Sparse Optimization Add Max {direction} {cnt + 1}/{num_2_add}")
                # Compute the score
                max_score = compute_score_list(max_list, self.initial_model)
                # Multiply the score by the multiplier
                max_score = max_score * max_mult
            else:
                max_score = np.inf
                print("Max Rectangles are out of bounds")
 

            # Adding the rectangles with min score
            if min_score >= max_score:
                for rect in max_list:
                    self.initial_rect_list.append(rect)
            else:
                for rect in min_list:
                    self.initial_rect_list.append(rect)

        print(f"Finished adding {num_2_add} {direction}(s)")
    
    def remove_rectangles(self, direction, num_2_remove):
        for _ in range(num_2_remove):
            # Find the current min, max rectangles
            min_list, max_list = find_next_rect(self.initial_rect_list, direction, edge = True)

            # Computing the scores
            min_score = compute_score_list(min_list, self.initial_model)
            max_score = compute_score_list(max_list, self.initial_model)

            if min_score <= max_score:
                self.initial_rect_list = remove_rectangles(self.initial_rect_list, max_list)
            else:
                self.initial_rect_list = remove_rectangles(self.initial_rect_list, min_list)
        
        print(f"Finished removing {num_2_remove} {direction}(s)")