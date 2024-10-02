import numpy as np
import cv2 as cv
import geopandas as gpd
from matplotlib import pyplot as plt
import os

from functions.image_processing import four_2_five_rect
from functions.rect_list import build_rect_list_points, set_id
from functions.display import disp_rectangles
from classes.model import model
import time
from functions.optimization import optimize_xy, optimize_t, optimize_hw
from functions.rect_list_processing import distance_optimize
from functions.general import create_shapefile

class optimize_plots:
    def __init__(self, plot_finder_job_params, loggers):
        self.params = plot_finder_job_params
        self.logger = loggers.optimize_plots
        self.phase_one()

    def phase_one(self):
        # Check that the shapefile exists
        shapefile_path = self.params["shapefile_path"]
        try:
            gdf = gpd.read_file(shapefile_path)
            if gdf is not None:
                self.logger.info(f"Shapefile found at: {shapefile_path}")
            else:
                self.logger.critical(f"Shapefile not found at: {shapefile_path}. Exiting...")
                exit(1)
        except Exception as e:
            self.logger.critical(f"Error reading shapefile at: {shapefile_path}. Error: {e}. Exiting...")
            exit(1)

        # Check that the shapefile has the correct CRS
        img_crs = self.params["meta_data"]['crs']
        original_transform = self.params["meta_data"]['transform']
        rotation_matrix = self.params["rotation_matrix"]

        gdf = gdf.to_crs(img_crs)

        # Looping through the shapefile
        rect_coords = []

        for indx, row in gdf.iterrows():
            # Extracting the coordinates of the boundary
            gps_coords = list(row.geometry.exterior.coords.xy)
            gps_coords = np.array(gps_coords, dtype = np.float64).T
            # Removing the last point as it is the same as the first
            gps_coords = gps_coords[:-1, :]
     
            # Converting to pixel coordinates
            pixel_coords = ~original_transform * gps_coords.T
            pixel_coords = np.array(pixel_coords).T

            if rotation_matrix is not None:
                pixel_coords = np.hstack((pixel_coords, np.ones((pixel_coords.shape[0], 1))))
                pixel_coords = np.dot(pixel_coords, rotation_matrix.T)
                

            pixel_coords = np.round(pixel_coords).astype(int)
            pixel_coords = np.flip(pixel_coords, axis = 1)

            #Sort the top and bottom points
            top_bottom = pixel_coords[np.argsort(pixel_coords[:,0]), :]
            top = top_bottom[:2, :]
            bottom = top_bottom[2:, :]

            # Sort left and right
            left_right_top = top[np.argsort(top[:,1]), :]
            left_right_bottom = bottom[np.argsort(bottom[:,1]), :]

            # Extracting the points
            top_left = left_right_top[0]
            top_right = left_right_top[1]
            bottom_left = left_right_bottom[0]
            bottom_right = left_right_bottom[1]

            # Creating the rectangle
            rect = four_2_five_rect([top_left, top_right, bottom_left, bottom_right])
            rect_coords.append(rect)

    
        # Get the required params to build the rectangles
        num_rows = self.params["number_rows"]
        num_ranges = self.params["number_ranges"]
        gray_img = self.params["gray_img"]

        initial_rect_list = build_rect_list_points(rect_coords, num_ranges, num_rows, gray_img, self.logger)

        # Get the params to set the ID's
        label_start = self.params["label_start"]
        label_flow = self.params["label_flow"]

        set_id(initial_rect_list, label_start, label_flow)

        self.phase_two(initial_rect_list)

    def phase_two(self, initial_rect_list):
        # Pull the params
        logger = self.logger
        model_size = self.params["model_size"]
        existing_models = self.params["models"]

        if existing_models:
            optimization_model = existing_models["initial_model"]
        else:
            optimization_model = model(model_size).compute_initial_model(initial_rect_list, self.logger)

        # Get the optimization params
        x_radi = self.params["x_radi"]
        y_radi = self.params["y_radi"]
        t_radi = self.params["t_radi"]
        shrink_width = self.params["shrink_width"]
        shrink_height = self.params["shrink_height"]

        iterations = self.params["optimization_iteration"]
        img = self.params["gray_img"]
        neighbor_radi = self.params["neighbor_radi"]
        kappa = 0.01

        optimized_rect_list = initial_rect_list

        for cnt in range(iterations):
            logger.info(f"Starting X,Y Optimization Iteration: {cnt + 1}")
            start_time = time.time()

            # Optimize XY
            optimized_rect_list = optimize_xy(optimized_rect_list, x_radi, y_radi, img, optimization_model)

            # Neighbor Optimization
            optimized_rect_list = distance_optimize(optimized_rect_list, neighbor_radi, kappa, logger)

            # Update the model
            optimization_model = model(model_size).compute_mean_model(optimized_rect_list)
            
            end_time = time.time()
            e_time = np.round(end_time - start_time, 2)
            logger.info(f"Finished Optimization Iteration: {cnt + 1} in {e_time} seconds")

        # Optimize Theta
        logger.info("Starting Theta Optimization")
        optimized_rect_list = optimize_t(optimized_rect_list, t_radi, optimization_model)

        # Shrink the rectangles
        logger.info("Starting Height and Width Optimization")
        optimized_rect_list = optimize_hw(optimized_rect_list, shrink_height, shrink_width, optimization_model)
        
        logger.info("Finished Optimization")
        # Create the output
        self.phase_three(optimized_rect_list)

    def phase_three(self, optimize_rect_list):
        logger = self.logger
        # Pull the params
        output_dir = self.params["pf_output_directorys"]
        shp_directory = output_dir["shapefiles"]
        img_name = self.params["image_name"]
        original_transform = self.params["meta_data"]["transform"] 
        original_crs = self.params["meta_data"]["crs"]
        inverse_rotation = self.params["inverse_rotation_matrix"]

        shp_path = os.path.join(shp_directory, f"{img_name}_optimized_plots.gpkg")
        logger.info(f"Creating Optimized Shapefile at: {shp_path}")
        create_shapefile(optimize_rect_list, original_transform, original_crs, inverse_rotation, shp_path)

        logger.info("Finished Optimizing Plots")
        return

