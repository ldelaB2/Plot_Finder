import numpy as np
import cv2 as cv
import geopandas as gpd
from matplotlib import pyplot as plt

from functions.image_processing import four_2_five_rect
from functions.rect_list import set_range_row, set_id
from functions.display import disp_rectangles

class optimize_plots:
    def __init__(self, plot_finder_job):
        self.pf_job = plot_finder_job
        self.pre_process()
        self.phase_one()

    def pre_process(self):
        # Pull the params
        nrows = self.pf_job.params['nrows']
        nranges = self.pf_job.params['nranges']
        label_start = self.pf_job.params["label_start"]
        label_flow = self.pf_job.params["label_flow"]
        shp_path = self.pf_job.params['shapefile_path']
        crs = self.pf_job.meta_data['crs']
        transform = self.pf_job.meta_data['transform']

        # Loading in the shapefile
        gdf = gpd.read_file(shp_path)

        # Converting to the same crs as the image
        gdf = gdf.to_crs(crs)
        
        # Looping through the shapefile
        rect_coords = []

        for indx, row in gdf.iterrows():
            # Extracting the coordinates of the boundary
            gps_coords = list(row.geometry.exterior.coords.xy)
            gps_coords = np.array(gps_coords, dtype = np.float64).T
            # Removing the last point as it is the same as the first
            gps_coords = gps_coords[:-1, :]
     
            # Converting to pixel coordinates
            pixel_coords = ~transform * gps_coords.T
            pixel_coords = np.round(pixel_coords).astype(int).T
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

        # Creating the rect list
        self.rect_list = build_rect_list(rect_coords, self.pf_job.img_ortho)
        set_range_row(self.rect_list, nranges, nrows)
        set_id(self.rect_list, label_start, label_flow)
        print("T")

       
    
    def phase_one(self):
        opt_param_dict = {}
        opt_param_dict['method'] = 'SA'
        opt_param_dict['x_radi'] = 20
        opt_param_dict['y_radi'] = 20
        opt_param_dict['theta_radi'] = 5
        opt_param_dict['maxiter'] = 100

        # Build the rectangles
        mean_width = np.mean([rect[2] for rect in self.rect_list])
        mean_height = np.mean([rect[3] for rect in self.rect_list])



        self.rect_list.build_model()
        self.rect_list.optimize_rectangles(opt_param_dict)
        tmp = self.rect_list.disp_rectangles()
        plt.imshow(tmp)

