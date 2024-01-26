import os, multiprocessing
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from functions import *
from wave_pad import wavepad
from sub_image import sub_image
from rectangles import rectangle
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
        except:
            pass

        try:
            os.mkdir(self.plots_path)
        except:
            pass

        try:
            os.mkdir(self.QC_path)
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

        return



    def phase2(self):
        fine_skip = [self.params["expand_radi"] * 2, self.params["expand_radi"] * 2]
        fine_grid, num_points = build_path(self.g_ortho.shape, self.params["box_radius"], fine_skip)
        
        with multiprocessing.Pool(processes=self.params["num_cores"]) as pool:
            rawwavepad = pool.map(
                compute_phase2_fun,
                [(self.params["freq_filter_width"], self.row_mask, self.range_mask, fine_grid[e], self.g_ortho, self.params["box_radius"], self.params["expand_radi"]) for e in range(num_points)])

        row_wavepad = np.ones_like(self.g_ortho).astype(np.float64)
        range_wavepad = np.ones_like(self.g_ortho).astype(np.float64)
        print("Finished processing fine grid")

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
        final_row_skel_output = working_wavepad.find_rows()
        final_range_skel_output = working_wavepad.find_ranges()

        # Saving the output for Quality Control
        if self.params["QC_depth"] != "none":
            name = 'Row_Skeleton.jpg'
            img = flatten_mask_overlay(self.rgb_ortho, final_row_skel_output, .5)
            img.save(os.path.join(self.QC_path, name))
            name = 'Range_Skeleton.jpg'
            img = flatten_mask_overlay(self.rgb_ortho, final_range_skel_output, .5)
            img.save(os.path.join(self.QC_path, name))
            print("Saved Skeletonized QC")
        
        # Finding the initial rectangles
        self.final_rect_list = working_wavepad.find_plots(self.g_ortho)

        # Saving the output for Quality Control
        if self.params["QC_depth"] != "none":
            name = 'FFT_Rectangles_Placement.jpg'
            img = disp_rectangles_img(self.final_rect_list, self.rgb_ortho, name = True)
            img.save(os.path.join(self.QC_path, name))
        
        print("Finished Computing FFT Rectangle Placement")

        return
        

    def optomize_plots(self):
        # Filtering out plots with low germination
        optomization_list, flagged_list = filter_rectangles(self.final_rect_list, self.rgb_ortho)

        # Saving the output for Quality Control
        if self.params["QC_depth"] != "none":
            name = 'Flagged_Rectangles.jpg'
            img = disp_rectangles_img(flagged_list, self.rgb_ortho)
            img.save(os.path.join(self.QC_path, name))

        # Setting optomization parameters
        x_radi = self.params["optomization_x_radi"]
        y_radi = self.params["optomization_y_radi"]
        theta_radi = self.params["optomization_theta_radi"]
        miter = self.params["optomization_miter"]
        meta_iter = self.params["optomization_meta_miter"]

        # Optomizing the rectangles
        for e in range(meta_iter):
            model = compute_model(optomization_list, self.rgb_ortho)
            for k in tqdm(range(len(optomization_list)), desc = f"Optomizaing Rectangles Iteration {e + 1}/{meta_iter}"):
                optomization_list[k].optomize_rectangle(self.rgb_ortho, model, x_radi, y_radi, theta_radi, miter)
           
            # Saving the output for Quality Control
            if self.params["QC_depth"] != "none" and e != meta_iter - 1:
                name = f'Optomized_Rectangles_Iteration_{e + 1}.jpg'
                img = disp_rectangles_img(optomization_list, self.rgb_ortho)
                img.save(os.path.join(self.QC_path, name))
                name = f'Optimization_Model_Iteration_{e + 1}.jpg'
                img = Image.fromarray(model)
                img.save(os.path.join(self.QC_path, name))
        
        print("Finished Optomizing Rectangles")
        self.final_rect_list = optomization_list + flagged_list

        if self.params["QC_depth"] != "none":
            name = 'Optomized_Plot_Placement.jpg'
            img = disp_rectangles_img(self.final_rect_list, self.rgb_ortho, name = True)
            img.save(os.path.join(self.QC_path, name))

        return
    
    def save_plots(self):
        with multiprocessing.Pool(processes=self.params["num_cores"]) as pool:
            pool.map(
                save_plots_fun,
                [(self.name, self.plots_path, rect, self.rgb_ortho) for rect in self.final_rect_list]
                )
        print("Finished Saving Plots")
        return
        

def save_plots_fun(args):
    img_name, plots_path, rect, img = args
    name = f'{img_name}_{rect.range}_{rect.row}.jpg'
    path = os.path.join(plots_path, name)
    rect.save_rect(path, img)


def compute_phase2_fun(args):
    FreqFilterWidth, row_mask, range_mask, center, image, boxradius, expand_radi = args
    subI = sub_image(image, boxradius, center)
    row_snip = subI.phase2(FreqFilterWidth, 0, row_mask, expand_radi)
    range_snip = subI.phase2(FreqFilterWidth, 1, range_mask, expand_radi)

    return (row_snip, range_snip, center)