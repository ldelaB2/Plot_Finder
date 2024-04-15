import os, multiprocessing
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import geopandas as gpd
import rasterio
from PIL import Image
from functions import *
from wave_pad import wavepad
from sub_image import sub_image
from tqdm import tqdm
from scipy.optimize import curve_fit

from sklearn.mixture import GaussianMixture


class ortho_photo:
    def __init__(self, params):
      print("T") 

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
            num_updated = 0
            for k in tqdm(range(len(optomization_list)), desc = f"Optomizaing Rectangles Iteration {e + 1}/{meta_iter}"):
                opt_flag = optomization_list[k].optomize_rectangle(self.rgb_ortho, model, x_radi, y_radi, theta_radi, miter)
                num_updated += opt_flag

            print(f"Improved {num_updated}/{len(optomization_list)} Rectangles")

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
    
    


