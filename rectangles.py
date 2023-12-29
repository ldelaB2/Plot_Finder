from operator import itemgetter
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numpy.fft import fft, fftshift

class rectangle:
    def __init__(self):
        self.rect_list = []
        self.img_rgb = None
        self.img_g = None

    def four_2_five_rect(self, points):
        top_left, top_right, bottom_left, bottom_right = points
        w1 = np.linalg.norm(top_left - top_right)
        w2 = np.linalg.norm(bottom_left - bottom_right)
        h1 = np.linalg.norm(top_left - bottom_left)
        h2 = np.linalg.norm(top_right - bottom_right)
        width = ((w1 + w2)/4).astype(int)
        height = ((h1 + h2)/4).astype(int)
        center = np.mean((top_left,top_right,bottom_left,bottom_right), axis = 0).astype(int)
        rect = (center, width, height, 0)
        self.rect_list.append(rect)

    def disp_rectangles(self):
        plt.close('all')
        fig, ax = plt.subplots(1)
        ax.imshow(self.img)

        for rect in self.rect_list:
            bottom_left = rect[0] - (rect[2], rect[1])
            bottom_left = bottom_left[::-1]
            rect_patch = patches.Rectangle(bottom_left, (rect[1] * 2), (rect[2] * 2), linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect_patch)
            
        ax.set_axis_off()
        ax.set_aspect(2)
        return fig
    
    def create_unit_square(self):
        self.unit_width = self.mean_width*2 + 1
        self.unit_height = self.mean_height*2 + 1
        # Creating the unit square
        y = np.linspace(-1, 1, self.unit_height)
        x = np.linspace(-1, 1, self.unit_width)
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
            extracted_img = np.reshape(extracted_img, (self.unit_height, self.unit_width, 1))
            
        return extracted_img
        
    def compute_fft_score(self, img):
        sig = np.sum(img, axis = 0)
        fsig = fftshift(fft(sig - np.mean(sig)))
        amp = abs(fsig)
        return amp
    
    def compute_train_set(self):
        train_set = []
        for rect in self.rect_list:
            img = self.extract_rectangle(rect, self.img_g)
            fsig = self.compute_fft_score(img)
            train_set.append(fsig)

        self.train_set = train_set
            
        

    def compute_stats(self):
        width = np.array(list(map(itemgetter(1), self.rect_list)))
        height = np.array(list(map(itemgetter(2), self.rect_list)))
        self.mean_width = np.mean(width).astype(int)
        self.mean_height = np.mean(height).astype(int)
    
    
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
                
                
               
                
                
            
    
        