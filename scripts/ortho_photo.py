import os, multiprocessing
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from functions import *
from wave_pad import wavepad
from sub_image import sub_image

class ortho_photo:
    def __init__(self, in_path, out_path, name):
        self.name = name[:-4]
        self.in_path = os.path.join(in_path, name)
        self.out_path = out_path
        self.create_output_dirs()

    def create_output_dirs(self):
        subdir = os.path.join(self.out_path, self.name)
        self.QC_path = os.path.join(subdir, "QC")
        self.plots_path = os.path.join(subdir, "plots")
        try:
            os.mkdir(subdir)
            os.mkdir(self.plots_path)
            os.mkdir(self.QC_path)
        except:
            pass

    def read_inphoto(self):
        self.rgb_ortho = cv.imread(self.in_path)
        self.create_g()
        self.rotate_img()

    def rotate_img(self):
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
        self.g_ortho = cv.cvtColor(self.rgb_ortho, cv.COLOR_BGR2LAB)[:,:,2]

    def build_scatter_path(self, boxradius, skip, expand_radi = None, disp = False):
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

    def build_wavepad(self, disp):
        self.row_wavepad = np.zeros(self.g_ortho.shape).astype(np.uint8)
        self.range_wavepad = np.zeros(self.g_ortho.shape).astype(np.uint8)

        for e in range(self.num_points):
            center = self.point_grid[e]
            expand_radi = self.expand_radi
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

    def find_plots(self, ncore):
        self.filtered_wavepad = wavepad(self.row_wavepad_binary, self.range_wavepad_binary, self.QC_path, ncore)
        self.compute_col_skel()
        self.compute_range_skel()
        self.find_rectangles()
        
    def find_rectangles(self):
        real_rect = self.filtered_wavepad.compute_rectangles()
        real_rect.img_rgb = self.rgb_ortho
        real_rect.img_g = self.g_ortho
        #rect_fig = real_rect.disp_rectangles()
        nrow = 96
        real_rect.impute_rows(nrow, self.filtered_wavepad.col_skel)

        
        #name = 'Real_Rectangles'
        #path = os.path.join(self.QC_path, name)
        #rect_fig.savefig(path, bbox_inches='tight', pad_inches=0, transparent=True)

    def compute_range_skel(self):
        real_range_output = self.filtered_wavepad.find_ranges()
        name = 'Real_Range_Skel.jpg'
        real_range_output = flatten_mask_overlay(self.rgb_ortho, real_range_output)
        real_range_output.save(os.path.join(self.QC_path, name))
        print("Saved Range Skeletonization QC")

    def compute_col_skel(self):
        real_col_output = self.filtered_wavepad.find_columns()
        imputed_col_output = self.filtered_wavepad.imput_col_skel()
        name = 'Real_Column_Skel.jpg'
        real_col_output = flatten_mask_overlay(self.rgb_ortho, real_col_output)
        real_col_output.save(os.path.join(self.QC_path, name))
        name = 'Imputed_Column_Skel.jpg'
        imputed_col_output = flatten_mask_overlay(self.rgb_ortho, imputed_col_output)
        imputed_col_output.save(os.path.join(self.QC_path, name))
        print("Saved Column Skeletonization QC")

    def phase1(self, FreqFilterWidth, num_sig_returned, vert_sig_remove, disp = False):
        #Find the signals
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
        self.row_mask = create_phase2_mask(row_sig, num_sig_returned, self.boxradius[1], vert_sig_remove, disp)
        self.range_mask = create_phase2_mask(range_sig, num_sig_returned, disp)
    
     
    def phase2(self, FreqFilterWidth, wave_pixel_expand, ncore = None):
        print(f"Using {ncore} cores to process fine grid")
        with multiprocessing.Pool(processes=ncore) as pool:
            rawwavepad = pool.map(
                compute_phase2_fun,
                [(FreqFilterWidth, self.row_mask, self.range_mask, wave_pixel_expand, self.point_grid[e], self.g_ortho, self.boxradius) for e in range(self.num_points)])

        self.rawwavepad = np.array(rawwavepad).reshape(-1,2)
        self.rawwavepad[:,0] = 1 - bindvec(self.rawwavepad[:,0])
        self.rawwavepad[:,1] = 1 - bindvec(self.rawwavepad[:,1])
        self.rawwavepad = (self.rawwavepad * 255).astype(np.uint8)


def compute_phase2_fun(args):
    FreqFilterWidth, row_mask, range_mask, num_pixles, center, image, boxradius = args
    raw_wavepad = np.zeros(2)
    subI = sub_image(image, boxradius, center)
    raw_wavepad[0] = subI.phase2(FreqFilterWidth, 0, row_mask, num_pixles)
    raw_wavepad[1] = subI.phase2(FreqFilterWidth, 1, range_mask, num_pixles)
    return (raw_wavepad)
