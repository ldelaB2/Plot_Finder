import os, multiprocessing
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from functions import *
from wave_pad import wavepad
from sub_image import sub_image
from rectangles import rectangle_list

class ortho_photo:
    def __init__(self, in_path, out_path, name):
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
        self.in_path = os.path.join(in_path, name)
        self.out_path = out_path
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
        subdir = os.path.join(self.out_path, self.name)
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
        self.rgb_ortho = cv.imread(self.in_path)
        self.create_g()
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

    def create_g(self):
        """
        This method creates a grayscale version of the input photo.

        It uses OpenCV's cvtColor function to convert the RGB image to LAB color space and then extracts the L channel (lightness) to create a grayscale image.

        Attributes Created:
            self.g_ortho: The grayscale version of the photo.
        """
        self.g_ortho = cv.cvtColor(self.rgb_ortho, cv.COLOR_BGR2LAB)[:,:,2]

    def build_scatter_path(self, boxradius, skip, expand_radi = None, disp = False):
        """
        This method builds a scatter path on the grayscale image.

        Parameters:
            boxradius (int): The radius of the box to be used in the scatter path.
            skip (tuple): The number of rows and columns to skip between points in the scatter path.
            expand_radi (tuple, optional): The number of rows and columns to expand the scatter path. If provided, it will override the skip parameter.
            disp (bool, optional): If True, it will display the scatter path on the grayscale image.

        Attributes Created:
            self.boxradius: The radius of the box to be used in the scatter path.
            self.point_grid: A list of points in the scatter path.
            self.num_points: The number of points in the scatter path.
        """
        if expand_radi is not None:
            skip = ((1 + 2 * expand_radi[0]), (1 + 2 * expand_radi[1]))
            self.expand_radi = expand_radi

        self.boxradius = boxradius
        img_shape = self.g_ortho.shape
        self.point_grid, self.num_points = build_path(img_shape, boxradius, skip)

        if disp:
            plt.imshow(self.g_ortho, cmap='gray')
            for point in self.point_grid:
                plt.scatter(point[0], point[1], c='red', marker='*')
                plt.axis('on')
            plt.show()

    def phase1(self, FreqFilterWidth, num_sig_returned, row_sig_remove, disp = False):
        """
        This method performs the first phase of the image processing.

        Parameters:
            FreqFilterWidth (int): The width of the frequency filter to be used.
            num_sig_returned (int): The number of signals to return.
            row_sig_remove (int): The the number of row signals to remove.
            disp (bool, optional): If True, it will display the average row and range signals.

        Attributes Created:
            self.row_mask: The mask created for the row signals.
            self.range_mask: The mask created for the range signals.
        """
        # Preallocate memory
        range_waves = np.zeros((self.num_points, (2 * self.boxradius[0])))
        row_waves = np.zeros((self.num_points, (2 * self.boxradius[1])))

        # Loop through sparse grid; returning the abs of Freq Wave
        for e in range(self.num_points):
            center = self.point_grid[e]
            subI = sub_image(self.g_ortho, self.boxradius, center)
            row_waves[e, :], range_waves[e, :] = subI.phase1(FreqFilterWidth)

        # Finding dominant frequency in row (column) direction
        row_sig = np.mean(row_waves, 0)
        # Finding dominant frequency in range (row) direction
        range_sig = np.mean(range_waves, 0)

        if disp:
            fig, axes = plt.subplots(nrows=1, ncols=2)
            axes[0].plot(row_sig)
            axes[0].set_title('Avg Row Signal')
            axes[1].plot(range_sig)
            axes[1].set_title('Avg Range Signal')
            plt.tight_layout()
            plt.show()
            
        # Creating the masks
        self.row_mask = create_phase2_mask(row_sig, num_sig_returned, self.boxradius[1], row_sig_remove, disp)
        self.range_mask = create_phase2_mask(range_sig, num_sig_returned, disp)

    def phase2(self, FreqFilterWidth, wave_pixel_expand, ncore = None):
        """
        This method performs the second phase of the wavepad creation process.

        Parameters:
            FreqFilterWidth (int): The width of the frequency filter.
            wave_pixel_expand (int): The number of pixels to expand the wavepad by.
            ncore (int, optional): The number of cores to use for parallel processing. Defaults to None.

        The method first creates a multiprocessing pool with the specified number of cores.
        It then maps the `compute_phase2_fun` function to the pool, passing in the frequency filter width, row and range masks, wave pixel expansion, point grid, ortho photo, and box radius for each point in the point grid.
        It reshapes the resulting raw wavepad and normalizes it to the range [0, 1].
        Finally, it multiplies the raw wavepad by 255 and converts it to an unsigned 8-bit integer.
        """
        with multiprocessing.Pool(processes=ncore) as pool:
            rawwavepad = pool.map(
                compute_phase2_fun,
                [(FreqFilterWidth, self.row_mask, self.range_mask, wave_pixel_expand, self.point_grid[e], self.g_ortho, self.boxradius) for e in range(self.num_points)])

        # Invert the wavepad because we want the signal to be maxamized inbetween rows/ ranges
        # The signal we find will always be what is white in the gray scale image
        self.rawwavepad = np.array(rawwavepad).reshape(-1,2)
        self.rawwavepad[:,0] = 1 - bindvec(self.rawwavepad[:,0])
        self.rawwavepad[:,1] = 1 - bindvec(self.rawwavepad[:,1])
        self.rawwavepad = (self.rawwavepad * 255).astype(np.uint8)

    def build_wavepad(self, disp):
        """
        This method builds the wavepad from the raw wavepad.

        Parameters:
            disp (bool, optional): Whether to display the wavepad. Defaults to False.

        The method first normalizes the raw wavepad to the range [0, 1].
        It then multiplies the normalized wavepad by 255 and converts it to an unsigned 8-bit integer.
        If the disp parameter is True, it displays the wavepad.
        Finally, it creates a wavepad object from the wavepad image and stores it in the wavepad attribute.
        """

        self.row_wavepad = np.zeros(self.g_ortho.shape).astype(np.uint8)
        self.range_wavepad = np.zeros(self.g_ortho.shape).astype(np.uint8)
        expand_radi = self.expand_radi

        for e in range(self.num_points):
            center = self.point_grid[e]
            rowstrt = center[1] - expand_radi[0]
            rowstp = center[1] + expand_radi[0] + 1
            colstrt = center[0] - expand_radi[1]
            colstp = center[0] + expand_radi[1] + 1

            self.row_wavepad[rowstrt:rowstp, colstrt:colstp] = self.rawwavepad[e, 0]
            self.range_wavepad[rowstrt:rowstp, colstrt:colstp] = self.rawwavepad[e, 1]

        # Saving the output for Quality Control
        name = 'Raw_Row_Wave.jpg'
        Image.fromarray(self.row_wavepad).save(os.path.join(self.QC_path, name))
        name = 'Range_Row_Wave.jpg'
        Image.fromarray(self.range_wavepad).save(os.path.join(self.QC_path, name))
        print("Saved Wavepad QC")
        self.filter_wavepad(disp)

    def filter_wavepad(self, disp):
        """
        This method filters the wavepad images.

        Parameters:
            disp (bool): Whether to display the filtered images.

        The method first applies Otsu's thresholding to the row and range wavepad images to create binary images.
        It then erodes the binary images using a 5x5 kernel with 3 iterations.
        It overlays the binary images on the RGB ortho photo and saves these images.
        If the disp parameter is True, it displays the overlay images.
        """
        _, self.row_wavepad_binary = cv.threshold(self.row_wavepad, 0, 1, cv.THRESH_OTSU)
        _, self.range_wavepad_binary = cv.threshold(self.range_wavepad, 0, 1, cv.THRESH_OTSU)

        kernel = np.ones((5,5), np.uint8)
        self.row_wavepad_binary = cv.erode(self.row_wavepad_binary, kernel, iterations=3)
        self.range_wavepad_binary = cv.erode(self.range_wavepad_binary, kernel, iterations=3)

        # Creating the QC output
        row_filtered_disp = flatten_mask_overlay(self.rgb_ortho, self.row_wavepad_binary, .5)
        range_filtered_disp = flatten_mask_overlay(self.rgb_ortho, self.range_wavepad_binary, .5)

        # Saving Output
        name = 'Row_Wavepad_Threshold_Disp.jpg'
        row_filtered_disp.save(os.path.join(self.QC_path, name))
        name = 'Range_Wavepad_Threshold_Disp.jpg'
        range_filtered_disp.save(os.path.join(self.QC_path, name))

        if disp:
            plt.imshow(row_filtered_disp)
            plt.show()
            plt.imshow(range_filtered_disp)
            plt.show()

    def find_train_plots(self, ncore, poly_degree_range, poly_degree_col):
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
        self.compute_train_plots()

    def compute_train_plots(self):
        starting_rect, range_cnt, row_cnt = build_rectangles(self.range_skel, self.col_skel)
        train_rect = rectangle_list(starting_rect, self.rgb_ortho)
        #train_rect.disp_rectangles(self.rgb_ortho)
        train_rect.optomize_placement()
        
        
    def find_all_plots(self, ncore, nrange, nrow):
        print("T")
        


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
    FreqFilterWidth, row_mask, range_mask, num_pixles, center, image, boxradius = args
    raw_wavepad = np.zeros(2)
    subI = sub_image(image, boxradius, center)
    raw_wavepad[0] = subI.phase2(FreqFilterWidth, 0, row_mask, num_pixles)
    raw_wavepad[1] = subI.phase2(FreqFilterWidth, 1, range_mask, num_pixles)
    return (raw_wavepad)
