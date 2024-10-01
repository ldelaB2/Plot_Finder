import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

from functions.image_processing import extract_rectangle, five_2_four_rect
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
        self.neighbor_position = {}
        self.recompute_unit_sqr = True
        self.x_radi = None
        self.y_radi = None
        self.template_img = None
        self.model_size = None

    def clear(self):
        self.added = False
        self.unit_sqr = None
        self.neighbors = None
        self.recompute_unit_sqr = True

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
    
    def compute_template_score_image(self):
        # Compute the top left corner
        corner_points = self.compute_corner_points()
        top_left = corner_points[-1]
        min_x = top_left[0] - self.x_radi
        min_y = top_left[1] - self.y_radi
        max_x = top_left[0] + self.x_radi
        max_y = top_left[1] + self.y_radi

        # Check that the values are valid
        if max_y < 0 or max_x < 0 or min_y > self.template_img.shape[0] or min_x > self.template_img.shape[1]:
            self.flagged = True
            self.template_score = None
            self.score = 0
            return
       
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(self.template_img.shape[1], max_x)
        max_y = min(self.template_img.shape[0], max_y)

        self.template_score = self.template_img[int(min_y):int(max_y), int(min_x):int(max_x)]
        self.score = np.min(self.template_score)

    def compute_template_position(self):
        if not self.flagged:  
            best_point = np.argwhere(self.template_score == self.score)[0]
            delta_x = best_point[1] - self.x_radi
            delta_y = best_point[0] - self.y_radi

            self.center_x += delta_x
            self.center_y += delta_y
        else:
            return
        
    
    def compute_neighbor_position(self, neighbor):
        rng_away = abs(self.range - neighbor[0])
        row_away = abs(self.row - neighbor[1])

        if neighbor[0] < self.range:
            rng_away *= -1
        if neighbor[1] < self.row:
            row_away *= -1

        dx = self.width * row_away
        dy = self.height * rng_away
        
        theta = np.radians(self.theta)

        dx_rot = dx * np.cos(theta) - dy * np.sin(theta)
        dy_rot = dx * np.sin(theta) + dy * np.cos(theta)

        dx_rot = int(np.round(dx_rot))
        dy_rot = int(np.round(dy_rot))

        estimated_x = self.center_x + dx_rot
        estimated_y = self.center_y + dy_rot

        return estimated_x, estimated_y
        


       
    
 
 
    