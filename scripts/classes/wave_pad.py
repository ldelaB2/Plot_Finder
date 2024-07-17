import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np

from functions.display import disp_rectangles
from functions.optimization import compute_model
from functions.wavepad import process_wavepad
from functions.rect_list import build_rect_list, sparse_optimize, final_optimize, set_range_row, set_id
from functions.rect_list_processing import remove_rectangles, add_rectangles, double_check


class wavepad:
    def __init__(self, range_wavepad, row_wavepad, params, img):
        self.params = params
        self.range_wavepad = range_wavepad
        self.row_wavepad = row_wavepad
        self.img = img

        self.phase_one() # Processing the wavepads into skels
        self.phase_two() # Building the initial rectangles and removing extra ranges/rows
        self.phase_three() # Finding the all ranges and rows
        self.phase_four() # Final Optimization and adding labels

    def phase_one(self):
        #Pull the params
        ncores = self.params["num_cores"]
        filter_method = self.params["wavepad_filter_method"]
        poly_deg_range = self.params["poly_deg_range"]
        poly_deg_row = self.params["poly_deg_row"]

        if ncores > 1:
            with mp.Pool( processes = 2) as pool:
                results = pool.map(process_wavepad, [[self.range_wavepad, filter_method, poly_deg_range, "range"], [self.row_wavepad, filter_method, poly_deg_row, "row"]])
            self.range_skel, self.row_skel = results
        else:
            self.range_skel = process_wavepad([self.range_wavepad, filter_method, poly_deg_range, "range"])
            self.row_skel = process_wavepad([self.row_wavepad, filter_method,poly_deg_row, "row"])

        print("Finished Processing Wavepads")

    def phase_two(self):
        # Pull the params
        range_cnt = self.params["nranges"]
        row_cnt = self.params["nrows"]

        # Finding the initial rectangles, range and row count
        initial_rect_list, initial_range_cnt, initial_row_cnt, initial_width, initial_height = build_rect_list(self.range_skel, self.row_skel, self.img)

        # Computing the number of ranges and rows to find
        self.ranges_2_find = range_cnt - initial_range_cnt
        self.rows_2_find = row_cnt - initial_row_cnt
        self.model_shape = (initial_height, initial_width)

        # Computing the initial model
        initial_model = compute_model(initial_rect_list, self.model_shape)
        # Creating the optimization param dict
        self.create_opt_param_dict("sparse")

        # Preform the first optimization
        sparse_optimize(initial_rect_list, initial_model, self.opt_param_dict)
        # Recompute the model
        self.initial_model = compute_model(initial_rect_list, self.model_shape)
        self.initial_rect_list = initial_rect_list
        
        print("Finished Building Initial Rectangles")
       
    def phase_three(self):
        # Add ranges
        if self.ranges_2_find > 0:
            print(f"Finding {self.ranges_2_find} missing range(s)")
            self.initial_rect_list = add_rectangles(self.initial_rect_list, "range", self.ranges_2_find, self.initial_model, self.opt_param_dict)

        # Add rows
        if self.rows_2_find > 0:
            print(f"Finding {self.rows_2_find} missing row(s)")
            self.initial_rect_list = add_rectangles(self.initial_rect_list, "row", self.rows_2_find, self.initial_model, self.opt_param_dict)

         # Remove the extra ranges if needed
        if self.ranges_2_find < 0:
            print(f"Removing {abs(self.ranges_2_find)} extra range(s) from FFT")
            self.initial_rect_list = remove_rectangles(self.initial_rect_list, "range", abs(self.ranges_2_find), self.initial_model)
            self.ranges_2_find = 0

        # Remove the extra rows if needed
        if self.rows_2_find < 0:
            print(f"Removing {abs(self.rows_2_find)} extra row(s) from FFT")
            self.initial_rect_list = remove_rectangles(self.initial_rect_list, "row", abs(self.rows_2_find), self.initial_model)
            self.rows_2_find = 0

        # Double check rows 
        print("Double Checking Rows")
        self.initial_rect_list = double_check(self.initial_rect_list, "row", self.initial_model, self.opt_param_dict)

        # Double check ranges
        print("Double Checking Ranges")
        self.initial_rect_list = double_check(self.initial_rect_list, "range", self.initial_model, self.opt_param_dict)

        print("Finished Finding All Rectangles")
        self.final_rect_list = self.initial_rect_list

    def phase_four(self):
        # Pulling the params
        label_start = self.params["label_start"]
        label_flow = self.params["label_flow"]

        # Setting the range and row
        set_range_row(self.final_rect_list)
        # Setting the id
        set_id(self.final_rect_list, start = label_start, flow = label_flow)
        print("Finished Adding Labels")

        # Final Optimization
        self.create_opt_param_dict("fine")
        final_optimize(self.final_rect_list, self.opt_param_dict, self.initial_model)


    def create_opt_param_dict(self, phase):
        opt_param_dict = {}
        opt_param_dict['optimization_loss'] = "L1"
        opt_param_dict['neighbor_radi'] = 2
        opt_param_dict['ncore'] = self.params["num_cores"]

        if phase == "sparse":
            opt_param_dict['x_radi'] = 20
            opt_param_dict['y_radi'] = 50
            opt_param_dict['valid_radi'] = 30
            opt_param_dict['nstart'] = 10

        elif phase == "fine":
            opt_param_dict['max_epoch'] = 10
            opt_param_dict['x_radi'] = 30
            opt_param_dict['y_radi'] = 100
            opt_param_dict['theta_radi'] = 5
            opt_param_dict['width_shrink'] = 20
            opt_param_dict['height_shrink'] = 80
            opt_param_dict['ntest_XY'] = 300
            opt_param_dict['ntest_HW'] = 100
            opt_param_dict['model_shape'] = self.model_shape
            opt_param_dict['preform_XY_optimization'] = True

        self.opt_param_dict = opt_param_dict

        return
