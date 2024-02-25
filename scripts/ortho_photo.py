import os, multiprocessing
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from PIL import Image
from functions import *
from wave_pad import wavepad
from sub_image import sub_image
from tqdm import tqdm
from scipy.optimize import curve_fit
from shapely.geometry import Polygon


class ortho_photo:
    def __init__(self, params):
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
        self.name = os.path.splitext(os.path.basename(params["input_path"]))[0]
        self.ortho_path = params["input_path"]
        self.params = params
        # Extract meta data
        with rasterio.open(self.ortho_path) as src:
            self.meta = src.meta

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
        # Calculating 2d FFT
        mean_subtracted = self.g_ortho - np.mean(self.g_ortho)
        rot_fft = np.fft.fft2(mean_subtracted)
        rot_fft = np.fft.fftshift(rot_fft)
        rot_fft = np.abs(rot_fft)
        #rot_fft = np.log(rot_fft + 1)

        # Creating the sub image
        box_radi = 30
        x_center = rot_fft.shape[1] // 2
        y_center = rot_fft.shape[0] // 2
        sub_image = rot_fft[y_center - box_radi:y_center + box_radi, x_center - box_radi:x_center + box_radi]

        # Creating the points
        points = np.linspace(-1, 1,box_radi * 2)
        points = np.column_stack((points, np.zeros(box_radi * 2), np.ones(box_radi * 2)))
        thetas = np.linspace(20, 160, 100)
        scores = []

        # Checking for the best angle
        for theta in thetas:
            theta = np.deg2rad(theta)
            aff_mat = create_affine_frame(box_radi, box_radi, theta, (box_radi - 1) * 2, (box_radi - 1) * 2)
            rot_points = np.dot(aff_mat, points.T).T
            rot_points = np.round(rot_points[:, :2], decimals = 0).astype(int)
            img_vals = sub_image[rot_points[:,0], rot_points[:,1]]
            img_vals = sorted(img_vals, reverse = True)
            temp_score = np.sum(img_vals[:8])
            scores.append(temp_score)

        # Finding the best angle
        opt_theta = thetas[np.argmax(scores)]
        theta = 90 - opt_theta
        print(f"Rotating the original Image by {theta:.1f} degrees")

        # Computing params for the inverse rotation matrix
        theta = np.deg2rad(theta)
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

        # Creating the inverse rotation matrix
        inverse_rotation_matrix = cv.getRotationMatrix2D((new_width/2,new_height/2), -theta, 1)
        inverse_rotation_matrix[0, 2] += (width - new_width) / 2
        inverse_rotation_matrix[1, 2] += (height - new_height) / 2
        self.inverse_rotation_matrix = inverse_rotation_matrix

        # Rotate the image
        self.g_ortho = cv.warpAffine(self.g_ortho, rotation_matrix, (new_width,new_height), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue = 0)
        self.rgb_ortho = cv.warpAffine(self.rgb_ortho, rotation_matrix, (new_width, new_height), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue = (0,0,0))

    def create_g(self, method):
        """
        This method creates a grayscale version of the input photo.

        It uses OpenCV's cvtColor function to convert the RGB image to LAB color space and then extracts the L channel (lightness) to create a grayscale image.

        Attributes Created:
            self.g_ortho: The grayscale version of the photo.
        """
        if method == 'PCA':
            # Bluring the image
            #iterations = 10
            #kernel = (5,5)
            #sigma = .5
            #tmp = self.rgb_ortho
            #for _ in range(iterations):
                #tmp = cv.GaussianBlur(tmp, kernel, sigma)
                
            # projecting pixels onto first principal component
            pixels = self.rgb_ortho.reshape(-1, 3)
            pixels_pca, _, _, pve = pca(pixels, 1)

            # Shifting data to be non-negative
            pixels_pca -= np.min(pixels_pca)
            
            # Comute the histogram of the PCA projected pixels
            bin_num = 1000
            pixels_hist, bin_edges = np.histogram(pixels_pca, bin_num, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Define initial parameters for the Gaussian fits
            p0_bimodal = [np.mean(pixels_pca), np.std(pixels_pca), np.mean(pixels_pca), np.std(pixels_pca), .5]
            params_bimodal, _ = curve_fit(bimodal_gaussian, bin_centers, pixels_hist, p0=p0_bimodal)

            # Create the binary mask
            prob_dist_1, prob_dist_2 = calculate_prob(pixels_pca, params_bimodal)
            segmented_img = np.where(prob_dist_1 > prob_dist_2, 1,0)
            segmented_img = segmented_img.reshape(self.rgb_ortho.shape[0], self.rgb_ortho.shape[1])

            # Check for inversion
            zero_cnt = np.sum(segmented_img == 0)
            one_cnt = np.sum(segmented_img == 1)
            if one_cnt > zero_cnt:
                segmented_img = 1 - segmented_img

            # Create the grayscale image
            g_ortho = np.copy(self.rgb_ortho)
            g_ortho[segmented_img == 0] = [0,0,0]
            self.g_ortho = cv.cvtColor(g_ortho, cv.COLOR_BGR2LAB)[:,:,2]

            # Normalizing the grayscale image
            self.g_ortho = self.g_ortho - np.min(self.g_ortho)
            self.g_ortho = self.g_ortho / np.max(self.g_ortho)
            self.g_ortho = (self.g_ortho * 255).astype(np.uint8)

            # Saving the output
            if self.params["QC_depth"] != "none":
                # Ploting the histogram and the bimodal Gaussian
                name = 'Histogram_and_Fitted_Bimodal_Gaussian.jpg'
                plt.hist(pixels_pca, bin_num, density=True, color = 'b', alpha = 0.7, label = 'Data')
                x = np.linspace(np.min(pixels_pca), np.max(pixels_pca), bin_num)
                plt.plot(x, bimodal_gaussian(x, *params_bimodal), 'r', lw=2, label = 'Bimodal Gaussian')
                #plt.plot(x, trimodal_gaussian(x, *params_trimodal), 'y', lw=2, label = 'Trimodal Gaussian')
                plt.title('Histogram and Fitted Bimodal Gaussian')
                plt.xlabel('Value')
                plt.ylabel('Density')
                plt.legend()
                plt.savefig(os.path.join(self.QC_path, name))
                plt.close()

                # Saving the gray scale image
                name = 'Gray_Scale_Image.jpg'
                Image.fromarray(self.g_ortho).save(os.path.join(self.QC_path, name))
            
    
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
        row_wavepad_binary = filter_wavepad(row_wavepad, method = 'otsu')
        range_wavepad_binary = filter_wavepad(range_wavepad, method = 'hist')
        
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
        optomization_list, flagged_list, model_list = filter_rectangles(self.final_rect_list, self.rgb_ortho)

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
            model = compute_model(model_list, self.rgb_ortho)
            for k in tqdm(range(len(optomization_list)), desc = f"Optomizaing Rectangles Iteration {e + 1}/{meta_iter}"):
                optomization_list[k].optomize_rectangle(self.rgb_ortho, model, x_radi, y_radi, theta_radi, miter)
           
            # Saving the output for Quality Control
            if self.params["QC_depth"] != "none" and e != meta_iter - 1:
                name = f'Optomized_Rectangles_Iteration_{e + 1}.jpg'
                img = disp_rectangles_img(optomization_list, self.rgb_ortho)
                img.save(os.path.join(self.QC_path, name))
                name = f'Optimization_Model_Iteration_{e + 1}.jpg'
                img = Image.fromarray(model.astype(np.uint8))
                img.save(os.path.join(self.QC_path, name))
        
        print("Finished Optomizing Rectangles")
        self.final_rect_list = optomization_list + flagged_list

        if self.params["QC_depth"] != "none":
            name = 'Optomized_Plot_Placement.jpg'
            img = disp_rectangles_img(self.final_rect_list, self.rgb_ortho, name = True)
            img.save(os.path.join(self.QC_path, name))
            name = f'Optimization_Model_Iteration_{e + 1}.jpg'
            img = Image.fromarray(model.astype(np.uint8))
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
        
    def create_shapefile(self):
        poly_data = []
        original_aff = self.meta['transform']
        original_crs = self.meta['crs']

        for rect in self.final_rect_list:
            points = rect.compute_corner_points()
            points = np.column_stack((points, np.ones(points.shape[0])))
            points = np.dot(self.inverse_rotation_matrix, points.T).T
            points = original_aff * points.T
            points = tuple(zip(points[0], points[1]))
            temp_poly = Polygon(points)
            poly_data.append({'geometry': temp_poly, 'label': rect.ID})
        
        gdf = gpd.GeoDataFrame(poly_data, crs=original_crs)
        file_name = self.params['output_path'] + f"/{self.name}" f"/{self.name}_plot_finder.gpkg"
        gdf.to_file(file_name, driver="GPKG")
        print("Finished Creating Shapefile")


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