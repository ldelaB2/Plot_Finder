from operator import itemgetter
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numpy.fft import fft, fftshift
from functions import eludian_distance, find_points, bindvec

class rectangle:
    def __init__(self, rect_list):
        self.rect_list = rect_list
        self.compute_stats()

    
    def create_unit_square(self):
        unit_width = self.mean_width*2 + 1
        unit_height = self.mean_height*2 + 1
        # Creating the unit square
        y = np.linspace(-1, 1, unit_height)
        x = np.linspace(-1, 1, unit_width)
        X, Y = np.meshgrid(x, y)
        unit_sqr = np.column_stack((X.ravel(), Y.ravel(), np.ones_like(X.ravel())))
        self.unit_sqr = unit_sqr

    def create_affine_frame(self, rect):
        center, width, height, theta = rect

        # Translation Matrix
        t_mat = np.zeros((3, 3))
        t_mat[0, 0], t_mat[1, 1], t_mat[2, 2] = 1, 1, 1
        t_mat[0, 2], t_mat[1, 2] = center[1], center[0]

        # Scaler Matrix
        s_mat = np.zeros((3, 3))
        s_mat[0, 0], s_mat[1, 1], s_mat[2, 2] = width, height, 1

        # Rotation Matrix
        r_1 = [np.cos(theta), np.sin(theta), 0]
        r_2 = [-np.sin(theta), np.cos(theta), 0]
        r_3 = [0, 0, 1]
        r_mat = np.column_stack((r_1, r_2, r_3))

        affine_mat = t_mat @ r_mat @ s_mat
        return affine_mat

    def extract_rectangle(self, rect, img):
        affine_mat = self.create_affine_frame(rect)
        rotated_points = np.dot(affine_mat, self.unit_sqr.T).T
        rotated_points = rotated_points.astype(int)
        
        z = img.shape
        if len(z) > 2:
            extracted_img = img[rotated_points[:, 1], rotated_points[:, 0], :]
            extracted_img = np.reshape(extracted_img, (self.unit_height, self.unit_width, img.shape[2]))
        else:
            extracted_img = img[rotated_points[:, 1], rotated_points[:, 0]]
            extracted_img = np.reshape(extracted_img, (self.unit_height, self.unit_width))
            
        return extracted_img
        
    
    def compute_fft_score(self, img):
        def compute_fft_score(img):
            sig = np.sum(img, axis = 0)
            fsig = fftshift(fft(sig - np.mean(sig)))
            amp = bindvec(abs(fsig))
            return amp

        scores = np.zeros((len(self.rect_list), self.mean_width*2 + 1))
        for e,rect in enumerate(self.rect_list):
            sub_img = self.extract_rectangle(rect, img)
            fsig = compute_fft_score(sub_img)
            scores[e,:] = fsig

        self.fft_scores = scores

    def compute_histogram_score(self, img):

        def compute_histogram(img):
            print("T")

        histogram = np.zeros((len(self.rect_list), self.mean_width*2 + 1))
        for e,rect in enumerate(self.rect_list):
            sub_img = self.extract_rectangle(rect, img)
            fsig = self.compute_fft_score(sub_img)
            scores[e,:] = fsig

        self.histograms = scores
            
    
    def compute_stats(self):
        width = np.array(list(map(itemgetter(1), self.rect_list)))
        height = np.array(list(map(itemgetter(2), self.rect_list)))
        self.mean_width = np.mean(width).astype(int)
        self.mean_height = np.mean(height).astype(int)
        self.create_unit_square()
    
    def optomize_placement(self, img):
        self.compute_histogram_score(img)
        sum_img = self.extract_rectangle(rect, img)


    def impute_rows(self, nrow, col_skel):
        self.compute_stats()
        self.create_unit_square()
        self.compute_train_set()

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
        dialated_skel = cv.dilate(col_skel, kernel)
        num_col, labeled_skel, _, _ = cv.connectedComponentsWithStats(dialated_skel.T)
        num_ranges = num_col - 1
        
        if num_ranges == nrow:
            print("No rows to impute")
            return
        elif num_ranges > nrow:
            print("To many rows found attempting to remove")
            return
        else:
            rows_to_impute = nrow - num_ranges
            tmp = np.copy(col_skel.T)
            indx = np.argwhere(tmp != 0)
            tmp[indx[:, 0], indx[:, 1]] = labeled_skel[indx[:, 0], indx[:, 1]]
            top_row = np.argwhere(tmp == 1)
            bottom_row = np.argwhere(tmp == num_ranges)
            cnt = 1
            
            while rows_to_impute >= cnt:
                top_extend = np.copy(top_row)
                top_extend[:,0]  = top_extend[:,0] - (2*self.mean_width + 1)
                
                
               
                
                
            
    
        