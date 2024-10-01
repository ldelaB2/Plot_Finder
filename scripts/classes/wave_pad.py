
class wavepad:
    def __init__(self, range_wavepad, row_wavepad, params, img):
        self.params = params
        self.range_wavepad = range_wavepad
        self.row_wavepad = row_wavepad
        self.img = img

        
        self.phase_two() # Building the initial rectangles and removing extra ranges/rows
        self.phase_three() # Finding the all ranges and rows
        self.phase_four() # Final Optimization and adding labels


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

        # Creating the optimization param dict
        self.create_opt_param_dict("sparse")

        # Initial model build by optimizing initial rect_list
        self.model = model(self.opt_param_dict)
        self.model.sparce_optimize(initial_rect_list, 20)

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
       



        return
