
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import dual_annealing, Bounds, minimize
from deap import base, creator, tools, algorithms
from pyswarm import pso
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
import cv2 as cv

from functions.image_processing import extract_rectangle, five_2_four_rect
from functions.optimization import compute_score
from functions.general import bindvec
from functions.image_processing import create_unit_square

class rectangle:
    def __init__(self, rect):
        self.center_y = rect[0]
        self.center_x = rect[1]
        self.width = rect[2]
        self.height = rect[3]
        self.theta = rect[4]

        if len(rect) == 7:
            self.range = rect[5]
            self.row = rect[6]
        else:
            self.range = None
            self.row = None

        self.img = None
        self.ID = None

        self.flagged = False
        self.added = False
        self.unit_sqr = None
        self.neighbors = None
        self.recompute_unit_sqr = True
        self.nbr_dxy = {}
        self.nbr_position = {}
        self.valid_points = None
        self.score = None

    def create_sub_image(self):
        if self.recompute_unit_sqr:
            self.unit_sqr = create_unit_square(self.width, self.height)
            self.recompute_unit_sqr = False

        sub_image = extract_rectangle(self.center_x, self.center_y, self.theta, self.width, self.height, self.unit_sqr, self.img)
        return sub_image
    
    def compute_corner_points(self):
        points = (self.center_x, self.center_y, self.width, self.height, self.theta)
        corner_points = five_2_four_rect(points)
        return corner_points
  
    def move_rectangle(self, dX, dY, dT):
        center_x = self.center_x + dX
        center_y = self.center_y + dY
        theta = self.theta + dT
        
        new_img = extract_rectangle(center_x, center_y, theta, self.width, self.height, self.unit_sqr, self.img)
        
        return new_img
    
    def shrink_rectangle(self, dwidth, dheight):
        new_width = self.width + dwidth
        new_height = self.height + dheight
        new_unit_sqr = create_unit_square(new_width, new_height)
        new_img = extract_rectangle(self.center_x, self.center_y, self.theta, new_width, new_height, new_unit_sqr, self.img)

        return new_img
    
 
    def clear(self):
        self.flagged = False
        self.added = False
        self.unit_sqr = None
        self.neighbors = None
        self.recompute_unit_sqr = True
        self.nbr_dxy = {}
        self.nbr_position = {}
        self.valid_points = None
        self.score = None

  
    def optimization_pre_process(self, x_radi, y_radi):
        self.x_radi = x_radi
        self.y_radi = y_radi

        # Compute the bounds
        self.center_x_bounds = [self.center_x - x_radi, self.center_x + x_radi]
        self.center_y_bounds = [self.center_y - y_radi, self.center_y + y_radi]
        self.center_x_bounds[0] = max(self.center_x_bounds[0], 0)
        self.center_y_bounds[0] = max(self.center_y_bounds[0], 0)
        self.center_x_bounds[1] = min(self.center_x_bounds[1], self.img.shape[1])
        self.center_y_bounds[1] = min(self.center_y_bounds[1], self.img.shape[0])

        # Compute the search image features
        self.compute_search_image(x_radi, y_radi)


    def compute_search_image(self, x_radi, y_radi):
        half_width = self.width // 2
        half_height = self.height // 2
        search_img_x_bound = [self.center_x - half_width - x_radi, self.center_x + half_width + x_radi]
        search_img_y_bound = [self.center_y - half_height -y_radi, self.center_y + half_height + y_radi]

        search_img_x_bound[0] = max(search_img_x_bound[0], 0)
        search_img_y_bound[0] = max(search_img_y_bound[0], 0)
        search_img_x_bound[1] = min(search_img_x_bound[1], self.img.shape[1])
        search_img_y_bound[1] = min(search_img_y_bound[1], self.img.shape[0])

        search_img = self.img[search_img_y_bound[0]:search_img_y_bound[1], search_img_x_bound[0]:search_img_x_bound[1]]
        self.search_img_shape = search_img.shape

        self.compute_search_img_features(search_img)

        return 

    def compute_search_img_features(self, search_img):
        orb = cv.ORB_create()
        kp, des = orb.detectAndCompute(search_img, None)
        self.search_img_features = {"kp": kp, "des": des}
        return
    
    def compute_feature_center(self, feature_dict, threshold):
        kp = self.search_img_features['kp']
        des = self.search_img_features['des']

        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
        match_count = [0] * len(kp)

        for key, value in feature_dict.items():
            matches = bf.match(des, value['descriptors'])
            for match in matches:
                match_count[match.queryIdx] += 1


        threshold = len(feature_dict) * threshold

        filtered_keypoints = []
        filtered_descriptors = []

        for i, count in enumerate(match_count):
            if count > threshold:
                filtered_keypoints.append(kp[i])
                filtered_descriptors.append(des[i])

        if len(filtered_keypoints) > 0:
            x = np.array([kp.pt[0] for kp in filtered_keypoints])
            y = np.array([kp.pt[1] for kp in filtered_keypoints])
            x = x.mean().astype(int)
            y = y.mean().astype(int)
            dx = x - self.search_img_shape[1] // 2
            dy = y - self.search_img_shape[0] // 2
        else:
            dx = 0
            dy = 0

        self.feat_dx = dx
        self.feat_dy = dy

        return
       
    def compute_template_center(self, template_img, x_offset, y_offset):
        points = self.compute_corner_points()
        top_left = points[-1]

        top_left[0] = top_left[0] - x_offset[0]
        top_left[1] = top_left[1] - y_offset[0]

        temp_search_x = [top_left[0] - self.x_radi, top_left[0] + self.x_radi]
        temp_search_y = [top_left[1] - self.y_radi, top_left[1] + self.y_radi]

        #temp_search_x[0] = max(temp_search_x[0], 0)
        #temp_search_x[1] = min(temp_search_x[1], template_img.shape[1])

        #temp_search_y[0] = max(temp_search_y[0], 0)
        #temp_search_y[1] = min(temp_search_y[1], template_img.shape[0])

        temp_search_img = template_img[temp_search_y[0]:temp_search_y[1], temp_search_x[0]:temp_search_x[1]]
        min_point = np.argwhere(temp_search_img == temp_search_img.min())[0]

        dy = min_point[0] - self.y_radi
        dx = min_point[1] - self.x_radi

        self.temp_dx = dx
        self.temp_dy = dy

        return


 
    