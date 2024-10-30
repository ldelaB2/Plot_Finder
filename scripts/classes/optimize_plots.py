import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt
import os
import time

from functions.image_processing import four_2_five_rect
from functions.rect_list import build_rect_list_points, set_id
from functions.display import save_model, save_results
from functions.optimization import compute_model, set_radi
from functions.rect_list_processing import distance_optimize
from functions.general import create_shapefile
from functions.optimization import compute_template_image
from tqdm import tqdm

class optimize_plots:
    def __init__(self, plot_finder_job_params, loggers):
        self.params = plot_finder_job_params
        self.logger = loggers.optimize_plots
        self.phase_zero()

    def phase_zero(self):
        shape_path = os.path.join(self.params["output_directory"], "shapefile")
        try: 
            paths = os.listdir(shape_path)
            for path in paths:
                if path.endswith(".gpkg"):
                    self.params["shapefile_path"] = os.path.join(shape_path, path)
                    break
        except:
            self.logger.critical(f"No shapefiles found at: {shape_path}. Exiting...")
            exit(1)
        
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
        if self.params["model_size"] is None:
            mean_width = np.mean([rect.width for rect in initial_rect_list])
            mean_height = np.mean([rect.height for rect in initial_rect_list])
            mean_width = np.round(mean_width).astype(int)
            mean_height = np.round(mean_height).astype(int)
            model_size = (mean_height, mean_width)
        else:
            model_size = self.params["model_size"]

        if self.params["models"] is None:
            optimization_model = compute_model(model_size, initial_rect_list, logger)
        else:
            optimization_model = self.params["models"]["initial_model"]
        
        # Save the initial model
        save_model(self.params, optimization_model, "initial_model", logger)

        optimized_rect_list = initial_rect_list

        # Get the optimization params
        x_radi = self.params["x_radi"]
        y_radi = self.params["y_radi"]
        t_radi = self.params["t_radi"]
        w_radi = self.params["width_radi"]
        h_radi = self.params["height_radi"]

        optimized_rect_list = set_radi(optimized_rect_list, x_radi, y_radi, t_radi, h_radi, w_radi)

        iterations = self.params["optimization_iteration"]
        img = self.params["gray_img"]
        neighbor_radi = self.params["neighbor_radi"]
        kappa = 0.01
        
        for cnt in range(iterations):
            logger.info(f"Starting Optimization Iteration: {cnt + 1}")
            start_time = time.time()

            template_img = compute_template_image(optimization_model, img)

            t_cnt = 0
            h_cnt = 0
            w_cnt = 0
            # Total Optimization
            for rect in tqdm(optimized_rect_list, desc = "Total Optimization"):
                rect.optimize_xy(template_img)
                t_cnt += rect.optimize_t(optimization_model, method = "L2")
                h_cnt += rect.optimize_height(optimization_model, method = "L2")
                w_cnt += rect.optimize_width(optimization_model, method = "L2")

            logger.info(f"Total Plots Updated: T: {t_cnt}, H: {h_cnt}, W: {w_cnt}")

            # Neighbor Optimization
            optimized_rect_list = distance_optimize(optimized_rect_list, model_size,  neighbor_radi, kappa, logger)

            # Update the model
            optimization_model = compute_model(model_size, optimized_rect_list, logger)

            # Save the model
            save_model(self.params, optimization_model, f"model_{cnt + 1}", logger)

            # Save the results
            save_results(self.params, [optimized_rect_list], [f"plots_optimized_{cnt + 1}"], "rect_list", logger)
            
            end_time = time.time()
            e_time = np.round(end_time - start_time, 2)
            logger.info(f"Finished Optimization Iteration: {cnt + 1} in {e_time} seconds")

   
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

        if self.params["save_plots"] == True:
            logger.info("Saving Plots")
            self.save_plots(optimize_rect_list)

        return
    
    def save_plots(self, rect_list):
        # Pull the params
        logger = self.logger
        output_dir = self.params["pf_output_directorys"]
        plots_dir = output_dir["plots"]
        img_ortho = self.params["img_ortho"]
        img_name = self.params["image_name"]
        
        logger.info(f"Saving Plots at: {plots_dir}")

        for rect in tqdm(rect_list, desc = "Saving Plots"):
            rect_path = os.path.join(plots_dir, f"{img_name}_plot_{rect.ID}.jpg")
            rect.save_rect(img_ortho, rect_path)

        logger.info("Finished Saving Plots")
        return

