import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

from functions.wavepad import filter_wavepad, trim_boarder, find_center_line, impute_skel
from functions.image_processing import find_correct_sized_obj
from functions.rectangle import set_range_row, check_within_img, compute_score_list, compare_next_to_current, set_id, build_rectangles, find_next_rect, remove_rectangles
from functions.optimization import build_rect_list, compute_model, sparse_optimize_list
from functions.display import disp_rectangles
from functions.optimization import compute_spiral_path, compute_neighbors, distance_optimize
import numpy as np
from functions.optimization import final_optimize_list


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
        self.initial_model = compute_model(self.initial_rect_list, initial= True)
        
        # Computing the number of ranges and rows to find
        self.ranges_2_find = range_cnt - initial_range_cnt
        self.rows_2_find = row_cnt - initial_row_cnt

        print("Finished Finding Center Lines and Initial Rectangles")

    def phase_three(self):
        if self.rows_2_find == 0 and self.ranges_2_find == 0:
            print("All plots found from FFT")

        # Remove ranges
        if self.ranges_2_find < 0:
            print(f"Removing {abs(self.ranges_2_find)} extra range(s) from FFT")
            self.remove_rectangles("range", abs(self.ranges_2_find))

        # Remove rows
        if self.rows_2_find < 0:
            print(f"Removing {abs(self.rows_2_find)} extra row(s) from FFT")
            self.remove_rectangles("row", abs(self.rows_2_find))
        
        # Optimize initial rectangles before adding
        self.sparse_optimize(self.initial_rect_list, txt = "Initial Optimization")
        # Recompute the model
        self.initial_model = compute_model(self.initial_rect_list, initial = False)

        # Add ranges
        if self.ranges_2_find > 0:
            print(f"Finding {self.ranges_2_find} missing range(s) from FFT")
            self.add_rectangles("range", self.ranges_2_find)
        
        # Add rows
        if self.rows_2_find > 0:
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
        set_range_row(self.final_rect_list)
        # Setting the id
        set_id(self.final_rect_list, start = label_start, flow = label_flow)
        
        self.phase_six()
        print("Finished Adding Labels")


    def sparse_optimize(self, rect_list, txt):
        opt_param_dict = {}
        opt_param_dict['method'] = 'quadratic'
        opt_param_dict['x_radi'] = 10
        opt_param_dict['y_radi'] = 10
        opt_param_dict['quadratic_num_points'] = 15
        opt_param_dict['optimization_loss'] = 'euclidean'
        opt_param_dict['threshhold'] = 3
        opt_param_dict['max_epoch'] = 10

        sparse_optimize_list(rect_list, self.initial_model, opt_param_dict, txt = txt)

        return

    def add_rectangles(self, direction, num_2_add):
        for cnt in range(num_2_add):
            # Find the next rectangles
            min_list, max_list = find_next_rect(self.initial_rect_list, direction, edge = False)

            # Check to make sure the rectangles are within the image
            min_mult, max_mult = check_within_img(min_list, max_list, self.img.shape)

            if min_mult != np.inf:
                # Optimize the rectangles
                self.sparse_optimize(min_list, txt = f"Sparse Optimization Add Min {direction} {cnt + 1}/{num_2_add}")
                # Compute the score
                min_score = compute_score_list(min_list, self.initial_model)
                # Multiply the score by the multiplier
                min_score = min_score * min_mult
            else:
                min_score = np.inf
                print("Min Rectangles are out of bounds")

            if max_mult != np.inf:
                # Optimize the rectangles
                self.sparse_optimize(max_list, txt = f"Sparse Optimization Add Max {direction} {cnt + 1}/{num_2_add}")
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




    def phase_six(self):
        # Define the kernel radi size
        kernel_radi = [2, 2]

        # Compute the spiral path
        spiral_path = compute_spiral_path(self.final_rect_list)

        # Compute the neighbors
        compute_neighbors(self.final_rect_list, kernel_radi)

        # Final Optimization
        meta_epoch = 3
        model_epoch = 2

        opt_param_dict = {}
        opt_param_dict['method'] = 'SA'
        opt_param_dict['x_radi'] = 10
        opt_param_dict['y_radi'] = 10
        opt_param_dict['theta_radi'] = 2
        opt_param_dict['optimization_loss'] = 'euclidean'
        opt_param_dict['maxiter'] = 250
        
        for meta_cnt in range(meta_epoch):
            print(f"Final Optimization Meta Epoch {meta_cnt + 1}/{meta_epoch}")
            # Apply the Distance optimization
            distance_optimize(self.final_rect_list, spiral_path)

            # Apply model optimization
            final_optimize_list(self.final_rect_list, self.initial_model, opt_param_dict, model_epoch)
            print("T")

        print("T")
