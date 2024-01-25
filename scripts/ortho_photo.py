import os, multiprocessing
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from functions import *
from wave_pad import wavepad
from sub_image import sub_image
from rectangles import rectangle
from operator import itemgetter
from copy import deepcopy
from tqdm import tqdm

class ortho_photo:
    def __init__(self, name, params):
        """
        This is the constructor method for the ortho_photo class.

        Parameters:
            in_path (str): The input directory where the image is located.
            out_path (str): The output directory where the outputs will be saved.
            name (str): The name of the photo file.

        Attributes Created:
            self.name: The name of the photo file without its extension.
            self.in_path: The full path to the photo file.
            self.out_path: The full path to the photo specific output directory.
            self.QC_path: The path to the "QC" subdirectory within the output directory.
            self.plots_path: The path to the "plots" subdirectory within the output directory.
        """
        self.name = os.path.splitext(name)[0]
        self.ortho_path = os.path.join(params["input_path"], name)
        self.params = params
        self.create_output_dirs()

    def create_output_dirs(self):
        """
        This method creates the necessary output directories for the ortho_photo object.

        It first constructs the path to the subdirectory within the output directory, using the name of the ortho_photo object.
        Then, it creates two subdirectories within this subdirectory: "QC" and "plots".

        If the directories already exist, it will not throw an error but simply pass.

        Attributes Created:
            self.QC_path: The path to the "QC" subdirectory within the output directory.
            self.plots_path: The path to the "plots" subdirectory within the output directory.
        """
        subdir = os.path.join(self.params["output_path"], self.name)
        self.QC_path = os.path.join(subdir, "QC")
        self.plots_path = os.path.join(subdir, "plots")
        try:
            os.mkdir(subdir)
            try:
                os.mkdir(self.plots_path)
            except:
                pass
            try:
                os.mkdir(self.QC_path)
            except:
                pass
        except:
            pass

    def read_inphoto(self):
        """
        This method reads the input photo into memory and performs initial processing.

        It first reads the photo from the file path stored in self.in_path using OpenCV's imread function, storing the result in self.rgb_ortho.
        Then, it calls the create_g method to create a grayscale version of the image.
        Finally, it calls the rotate_img method to rotate the image.

        Attributes Created:
            self.rgb_ortho: The original photo read into memory as an array.
        """
        self.rgb_ortho = cv.imread(self.ortho_path)
        self.create_g(self.params["gray_scale_method"])
        self.rotate_img()

    def rotate_img(self):
        """
        This method rotates the grayscale and RGB versions of the image by a specified angle.

        It first calculates the rotation matrix for the specified angle using OpenCV's getRotationMatrix2D function.
        Then, it calculates the dimensions of the new image that will contain all the original image after rotation.
        It adjusts the translation part of the rotation matrix to prevent cropping of the image.
        Finally, it applies the rotation to the grayscale and RGB versions of the image using OpenCV's warpAffine function.

        Attributes Modified:
            self.g_ortho: The grayscale version of the photo, rotated counter clockwise by the specified angle.
            self.rgb_ortho: The RGB version of the photo, rotated counter clockwise by the specified angle.
        """
        theta = 1
        
        height, width = self.g_ortho.shape[:2]
        rotation_matrix = cv.getRotationMatrix2D((width/2,height/2), theta, 1)

        # Determine the size of the rotated image
        cos_theta = np.abs(rotation_matrix[0, 0])
        sin_theta = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin_theta) + (width * cos_theta))
        new_height = int((height * cos_theta) + (width * sin_theta))

        # Adjust the translation in the rotation matrix to prevent cropping
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2
        
        # Rotate the image
        self.g_ortho = cv.warpAffine(self.g_ortho, rotation_matrix, (new_width,new_height))
        self.rgb_ortho = cv.warpAffine(self.rgb_ortho, rotation_matrix, (new_width, new_height))

    def create_g(self, method):
        """
        This method creates a grayscale version of the input photo.

        It uses OpenCV's cvtColor function to convert the RGB image to LAB color space and then extracts the L channel (lightness) to create a grayscale image.

        Attributes Created:
            self.g_ortho: The grayscale version of the photo.
        """
        if method == "LAB":
            self.g_ortho = cv.cvtColor(self.rgb_ortho, cv.COLOR_BGR2LAB)[:,:,2]

    
    def phase1(self):
        FreqFilterWidth = self.params["freq_filter_width"]
        num_sig_returned = self.params["num_sig_returned"]
        row_sig_remove = self.params["row_sig_remove"]
        
        sparse_grid, num_points = build_path(self.g_ortho.shape, self.params["box_radius"], self.params["sparse_skip"])
        
        # Preallocate memory
        range_waves = np.zeros((num_points, (2 * self.params["box_radius"][0])))
        row_waves = np.zeros((num_points, (2 * self.params["box_radius"][1])))

        # Loop through sparse grid; returning the abs of Freq Wave
        for e in range(num_points):
            subI = sub_image(self.g_ortho, self.params["box_radius"], sparse_grid[e])
            row_waves[e, :], range_waves[e, :] = subI.phase1(FreqFilterWidth)

        # Finding dominant frequency in row (column) direction
        row_sig = np.mean(row_waves, 0)
        # Finding dominant frequency in range (row) direction
        range_sig = np.mean(range_waves, 0)

        if self.params["row_sig_remove"] is not None:
            row_sig[(self.params["box_radius"][1] - self.params["row_sig_remove"]):(self.params["box_radius"][1] + self.params["row_sig_remove"])] = 0
            
        # Creating the masks
        self.row_mask = create_phase2_mask(row_sig, num_sig_returned)
        self.range_mask = create_phase2_mask(range_sig, num_sig_returned)

        # Saving the output for Quality Control
        if self.params["QC_depth"] == "max":
            name = 'Phase2_Row_Mask.jpg'
            fig, _ = plot_mask(self.row_mask)
            fig.savefig(os.path.join(self.QC_path, name))

            name = 'Phase2_Range_Mask.jpg'
            fig, _ = plot_mask(self.range_mask)
            fig.savefig(os.path.join(self.QC_path, name))



    def phase2(self):
        fine_skip = [self.params["expand_radi"] * 2, self.params["expand_radi"] * 2]
        fine_grid, num_points = build_path(self.g_ortho.shape, self.params["box_radius"], fine_skip)
        
        with multiprocessing.Pool(processes=self.params["num_cores"]) as pool:
            rawwavepad = pool.map(
                compute_phase2_fun,
                [(self.params["freq_filter_width"], self.row_mask, self.range_mask, fine_grid[e], self.g_ortho, self.params["box_radius"], self.params["expand_radi"]) for e in range(num_points)])

        # Invert the wavepad because we want the signal to be maxamized inbetween rows/ ranges
        # The signal we find will always be what is white in the gray scale image
        row_wavepad = np.ones_like(self.g_ortho).astype(np.float64)
        range_wavepad = np.ones_like(self.g_ortho).astype(np.float64)

        for e in range(len(rawwavepad)):
            center = rawwavepad[e][2]
            col_min = center[0] - self.params["expand_radi"]
            col_max = center[0] + self.params["expand_radi"] + 1
            row_min = center[1] - self.params["expand_radi"]
            row_max = center[1] + self.params["expand_radi"] + 1

            row_snp = np.tile(rawwavepad[e][0], (self.params["expand_radi"] * 2 + 1, 1))
            range_snp = np.tile(rawwavepad[e][1], (self.params["expand_radi"] * 2 + 1, 1)).T

            row_wavepad[row_min:row_max, col_min:col_max] = row_snp
            range_wavepad[row_min:row_max, col_min:col_max] = range_snp


        row_wavepad = 1 - row_wavepad
        range_wavepad = 1 - range_wavepad
        row_wavepad = (row_wavepad * 255).astype(np.uint8)
        range_wavepad = (range_wavepad * 255).astype(np.uint8)

        # Saving the output for Quality Control
        if self.params["QC_depth"] == "max":
            name = 'Row_Wavepad_Raw.jpg'
            Image.fromarray(row_wavepad).save(os.path.join(self.QC_path, name))
            name = 'Range_Wavepad_Raw.jpg'
            Image.fromarray(range_wavepad).save(os.path.join(self.QC_path, name))
            print("Saved Raw Wavepad QC")

        # Filtering to create binary wavepads
        row_wavepad_binary = filter_wavepad(row_wavepad, 3)
        range_wavepad_binary = filter_wavepad(range_wavepad, 3)
        
        # Saving the output for Quality Control
        if self.params["QC_depth"] != "none":
            name = 'Row_Wavepad_Binary.jpg'
            img = flatten_mask_overlay(self.rgb_ortho, row_wavepad_binary, .5)
            img.save(os.path.join(self.QC_path, name))
            name = 'Range_Wavepad_Binary.jpg'
            img = flatten_mask_overlay(self.rgb_ortho, range_wavepad_binary, .5)
            img.save(os.path.join(self.QC_path, name))
            print("Saved Thresholded Wavepad QC")

        # Creating the wavepad object
        working_wavepad = wavepad(row_wavepad_binary, range_wavepad_binary, self.QC_path, self.params)
        working_wavepad.find_rows()
        working_wavepad.find_ranges()
        
        
      

    def find_plots(self, ncore, poly_degree_range, poly_degree_col, nrange, nrow):
        """
        This method finds the plots in the wavepad image.

        Parameters:
            ncore (int): The number of cores to use for parallel processing.
            poly_degree_range (int): The degree of the polynomial to fit to the mean 'y' values for each 'x' in the range skeleton image.
            poly_degree_col (int): The degree of the polynomial to fit to the mean 'y' values for each 'x' in the column skeleton image.

        The method first finds the ranges and columns in the wavepad image using the specified polynomial degrees.
        It then imputes missing columns in the column skeleton image.
        It finds the intersections of the range and column skeleton images to find the corner points of rectangles.
        """
        self.filtered_wavepad = wavepad(self.row_wavepad_binary, self.range_wavepad_binary, self.QC_path, ncore)
        self.compute_col_skel(poly_degree_col)
        self.compute_range_skel(poly_degree_range)
        
        starting_rect, range_cnt, row_cnt = build_rectangles(self.range_skel, self.col_skel)

        mean_width = np.mean(np.array(list(map(itemgetter(2), starting_rect)))).astype(int)
        mean_height = np.mean(np.array(list(map(itemgetter(3), starting_rect)))).astype(int)

        rows_2_find = nrow - row_cnt
        ranges_2_find = nrange - range_cnt

        for e, rect in enumerate(starting_rect):
            rect = rectangle(rect)
            rect.width = mean_width
            rect.height = mean_height
            starting_rect[e] = rect

        if rows_2_find > 0 or ranges_2_find > 0:
            self.impute_rectangles(starting_rect, rows_2_find, ranges_2_find)
        elif rows_2_find < 0 or ranges_2_find < 0:
            self.remove_rectangles(starting_rect, rows_2_find, ranges_2_find)
        else:
            print("No Rectangles to Impute or Remove")
            self.final_rect_list = starting_rect


    def optomize_plots(self, miter, x_radi, y_radi, theta_radi):
        for e in range(miter):
            print(f"Starting Optomization Iteration {e + 1}")
            model = compute_model(self.final_rect_list, self.rgb_ortho)
            for k in tqdm(range(len(self.final_rect_list)), desc = "Optomizing Rectangles"):
                self.final_rect_list[k].optomize_rectangle(self.rgb_ortho, model, x_radi, y_radi, theta_radi)
           
            disp_rectangles(self.final_rect_list, self.rgb_ortho)
        
        print("Finished Optomizing Rectangles")


    def impute_rectangles(self, starting_rect, rows_2_find, ranges_2_find):
        train_fft_scores = compute_fft_mat(starting_rect, self.g_ortho)
        d_height = starting_rect[0].height
        d_width = starting_rect[0].width

        # Finding missing ranges
        while ranges_2_find > 0:
            ranges = [rect.range for rect in starting_rect]
            max_range = np.max(ranges)
            min_range = np.min(ranges)

            top_rect = np.argwhere(ranges == min_range)
            bottom_rect = np.argwhere(ranges == max_range)
            temp_top_list = []
            temp_bottom_list = []
            for e in range(len(top_rect)):
                temp_top_rect = deepcopy(starting_rect[top_rect[e,0]])
                temp_bottom_rect = deepcopy(starting_rect[bottom_rect[e,0]])
                temp_top_rect.center_y = temp_top_rect.center_y - d_height
                temp_bottom_rect.center_y = temp_bottom_rect.center_y + d_height
                temp_top_rect.range = min_range - 1
                temp_bottom_rect.range = max_range + 1
                temp_top_list.append(temp_top_rect)
                temp_bottom_list.append(temp_bottom_rect)

            top_fft_scores = compute_fft_mat(temp_top_list, self.g_ortho)
            bottom_fft_scores = compute_fft_mat(temp_bottom_list, self.g_ortho)

            top_dist = compute_fft_distance(top_fft_scores, train_fft_scores)
            bottom_dist = compute_fft_distance(bottom_fft_scores, train_fft_scores)
            if top_dist < bottom_dist:
                starting_rect = starting_rect + temp_top_list
            else:
                starting_rect = starting_rect + temp_bottom_list
            
            ranges_2_find -= 1
        
        # Finding missing rows
        while rows_2_find > 0:
            rows = [rect.row for rect in starting_rect]
            max_row = np.max(rows)
            min_row = np.min(rows)

            left_rect = np.argwhere(rows == min_row)
            right_rect = np.argwhere(rows == max_row)
            temp_left_list = []
            temp_right_list = []
            for e in range(len(left_rect)):
                temp_left_rect = deepcopy(starting_rect[left_rect[e,0]])
                temp_right_rect = deepcopy(starting_rect[right_rect[e,0]])
                temp_left_rect.center_x = temp_left_rect.center_x - d_width
                temp_right_rect.center_x = temp_right_rect.center_x + d_width
                temp_left_rect.row = min_row - 1
                temp_right_rect.row = max_row + 1
                temp_left_list.append(temp_left_rect)
                temp_right_list.append(temp_right_rect)
            
            left_fft_scores = compute_fft_mat(temp_left_list, self.g_ortho)
            right_fft_scores = compute_fft_mat(temp_right_list, self.g_ortho)

            left_dist = compute_fft_distance(left_fft_scores, train_fft_scores)
            right_dist = compute_fft_distance(right_fft_scores, train_fft_scores)

            if left_dist < right_dist:
                starting_rect = starting_rect + temp_left_list
            else:
                starting_rect = starting_rect + temp_right_list

            rows_2_find -= 1

        self.final_rect_list = starting_rect
        print("Finished Imputing Edge Rectangles")


    def compute_range_skel(self, poly_degree):
        """
        This method computes the range skeleton of the wavepad image.

        Parameters:
            poly_degree (int): The degree of the polynomial to fit to the mean 'y' values for each 'x' in the skeleton image.

        The method first finds the ranges in the wavepad image using the specified polynomial degree.
        It overlays the real range skeleton image on the RGB ortho photo and saves this image.
        Finally, it prints a message indicating that the range skeletonization QC has been saved.
        """
        real_range_output, self.range_skel = self.filtered_wavepad.find_ranges(poly_degree)
        name = 'Real_Range_Skel.jpg'
        real_range_output = flatten_mask_overlay(self.rgb_ortho, real_range_output)
        real_range_output.save(os.path.join(self.QC_path, name))
        print("Saved Range Skeletonization QC")

    def compute_col_skel(self, poly_degree):
        """
        This method computes the column skeleton of the wavepad image.

        Parameters:
            poly_degree (int): The degree of the polynomial to fit to the mean 'y' values for each 'x' in the skeleton image.

        The method first finds the columns in the wavepad image using the specified polynomial degree.
        It then imputes missing columns in the column skeleton image.
        It overlays the real and imputed column skeleton images on the RGB ortho photo and saves these images.
        Finally, it prints a message indicating that the column skeletonization QC has been saved.
        """
        real_col_output = self.filtered_wavepad.find_columns(poly_degree)
        imputed_col_output, self.col_skel = self.filtered_wavepad.imput_col_skel()
        name = 'Real_Column_Skel.jpg'
        real_col_output = flatten_mask_overlay(self.rgb_ortho, real_col_output)
        real_col_output.save(os.path.join(self.QC_path, name))
        name = 'Imputed_Column_Skel.jpg'
        imputed_col_output = flatten_mask_overlay(self.rgb_ortho, imputed_col_output)
        imputed_col_output.save(os.path.join(self.QC_path, name))
        print("Saved Column Skeletonization QC")

   
    
def compute_phase2_fun(args):
    """
    This function computes the phase 2 of the wavepad creation process for a given set of arguments.

    Parameters:
        args (tuple): A tuple containing the following parameters:
            - FreqFilterWidth (int): The width of the frequency filter.
            - row_mask (ndarray): The row mask.
            - range_mask (ndarray): The range mask.
            - num_pixles (int): The number of pixels to be used in the calculation of the pixel value.
            - center (tuple): The center of the sub image.
            - image (ndarray): The ortho photo.
            - boxradius (int): The radius of the box.

    Returns:
        tuple: A tuple containing the raw wavepad for the row and range masks.

    The function first extracts the sub image from the ortho photo using the specified box radius and center.
    It then computes the phase 2 of the wavepad creation process for the row and range masks, passing in the frequency filter width and number of pixels to expand the wavepad by.
    Finally, it returns a tuple containing the raw wavepad for the row and range masks.
    """
    FreqFilterWidth, row_mask, range_mask, center, image, boxradius, expand_radi = args
    subI = sub_image(image, boxradius, center)
    row_snip = subI.phase2(FreqFilterWidth, 0, row_mask, expand_radi)
    range_snip = subI.phase2(FreqFilterWidth, 1, range_mask, expand_radi)

    return (row_snip, range_snip, center)