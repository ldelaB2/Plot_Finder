from operator import itemgetter
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numpy.fft import fft, fftshift
from functions import  bindvec
from scipy.optomize import minimize

class rectangle:
    def __init__(self, rect):
        self.center = rect[0]
        self.width = rect[1]
        self.height = rect[2]
        self.theta = rect[3]
        self.points = None
        self.range = rect[4][0]
        self.row = rect[4][1]
        self.red_histogram = None
        self.green_histogram = None
        self.blue_histogram = None
        self.hist_score = None
        self.fft_score = None


class rectangle_list:
    def __init__(self, rect_list, rgb_img):
        self.rect_list = rect_list
        self.compute_stats()
        self.rgb_img = rgb_img
        
        for e, rect in enumerate(self.rect_list):
            rect = rectangle(rect)
            self.compute_points(rect)
            self.compute_histogram(rect)
            self.rect_list[e] = rect

    
    def compute_stats(self):
        width = np.array(list(map(itemgetter(1), self.rect_list)))
        height = np.array(list(map(itemgetter(2), self.rect_list)))
        self.mean_width = np.mean(width).astype(int)
        self.mean_height = np.mean(height).astype(int)
        self.create_unit_square()  


    def create_unit_square(self):
        self.unit_width = self.mean_width*2 + 1
        self.unit_height = self.mean_height*2 + 1
        # Creating the unit square
        y = np.linspace(-1, 1, self.unit_height)
        x = np.linspace(-1, 1, self.unit_width)
        X, Y = np.meshgrid(x, y)
        unit_sqr = np.column_stack((X.ravel(), Y.ravel(), np.ones_like(X.ravel())))
        self.unit_sqr = unit_sqr

    
    def create_affine_frame(self, center, theta):
        width = self.unit_width
        height = self.unit_height

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
    
    def compute_points(self, rect):
        affine_mat = self.create_affine_frame(rect.center, rect.theta)
        rotated_points = np.dot(affine_mat, self.unit_sqr.T).T
        rotated_points = rotated_points.astype(int)
        rect.points = rotated_points[:,:2]

    def compute_histogram(self, rect):
        sub_image = self.extract_rectangle(rect)

        # Compute the histogram for each channel
        red_histogram = np.histogram(sub_image[:,:,0], bins=256, range=(0, 256))
        green_histogram = np.histogram(sub_image[:,:,1], bins=256, range=(0, 256))
        blue_histogram = np.histogram(sub_image[:,:,2], bins=256, range=(0, 256))

        rect.red_histogram = red_histogram
        rect.green_histogram = green_histogram
        rect.blue_histogram = blue_histogram

    def disp_histogram(self, rect):
        red_histogram = rect.red_histogram
        green_histogram = rect.green_histogram
        blue_histogram = rect.blue_histogram

        # Plot the histogram
        plt.figure(figsize=(10,6))
        plt.plot(red_histogram[1][:-1], red_histogram[0], color='red')
        plt.plot(green_histogram[1][:-1], green_histogram[0], color='green')
        plt.plot(blue_histogram[1][:-1], blue_histogram[0], color='blue')
        plt.title("RGB Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.show()


    def extract_rectangle(self, rect):
        rotated_points = rect.points
        img  = self.rgb_img
        z = img.shape
        if len(z) > 2:
            extracted_img = img[rotated_points[:, 1], rotated_points[:, 0], :]
            extracted_img = np.reshape(extracted_img, (self.unit_height, self.unit_width, img.shape[2]))
        else:
            extracted_img = img[rotated_points[:, 1], rotated_points[:, 0]]
            extracted_img = np.reshape(extracted_img, (self.unit_height, self.unit_width))
            
        return extracted_img
        
    def optomize_placement(self):
        normalized_hist_dist = np.zeros((len(self.rect_list), len(self.rect_list)))
        for e in range(len(self.rect_list)):
            for f in range(len(self.rect_list)):
                normalized_hist_dist[e,f] = self.compute_hist_dist(self.rect_list[e], self.rect_list[f])

        normalized_dist_sum = np.sum(normalized_hist_dist, axis = 0)
        normalized_mean = np.mean(normalized_dist_sum).astype(int)
        train_set = np.argwhere(normalized_dist_sum <= normalized_mean)
        bad_set = np.argwhere(normalized_dist_sum > normalized_mean)
        self.comptue_train_matrix(train_set)

        for indx in bad_set:
            self.compute_hist_score(self.rect_list[indx[0]])
        


        plt.plot(normalized_dist_sum)
        plt.axhline(y = normalized_mean, color = 'r')
            
    def error_function(self, new_rect, starting_score):
        self.compute_points(new_rect)
        self.compute_histogram(new_rect)
        self.compute_hist_score(new_rect)
        new_score = new_rect.hist_score
        error = new_score - starting_score
        return error

    def compute_hist_score(self, rect):
        tmp_mat = np.zeros_like(self.train_matrix)
        tmp_mat[:,:,0] = rect.red_histogram[0]
        tmp_mat[:,:,1] = rect.blue_histogram[0]
        tmp_mat[:,:,2] = rect.green_histogram[0]
        raw_dist = tmp_mat - self.train_matrix
        dist = np.linalg.norm(raw_dist).astype(int)
        rect.hist_score = dist

    def comptue_train_matrix(self, train_set):
        train_matrix = np.zeros((train_set.size, 256, 3))
        for indx, rect_indx in enumerate(train_set):
            rect = self.rect_list[rect_indx[0]]
            train_matrix[indx,:,0] = rect.red_histogram[0]
            train_matrix[indx,:,1] = rect.blue_histogram[0]
            train_matrix[indx,:,2] = rect.green_histogram[0]
        
        self.train_matrix = train_matrix
        
    def compute_hist_dist(self, rect1, rect2):
        red_hist1 = rect1.red_histogram[0]
        green_hist1 = rect1.green_histogram[0]
        blue_hist1 = rect1.blue_histogram[0]
        
        red_hist2 = rect2.red_histogram[0]
        green_hist2 = rect2.green_histogram[0]
        blue_hist2 = rect2.blue_histogram[0]
        
        red_dist = np.linalg.norm(red_hist1 - red_hist2)
        green_dist = np.linalg.norm(green_hist1 - green_hist2)
        blue_dist = np.linalg.norm(blue_hist1 - blue_hist2)
        
        normalized_hist_dist = ((red_dist + green_dist + blue_dist) / 3).astype(int)
        
        return normalized_hist_dist

    def disp_rectangles(self, img):
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        for rect in self.rect_list:
            center_x, center_y, width, height = rect.center[1], rect.center[0], rect.width, rect.height
            bottom_left_x = center_x - width
            bottom_left_y = center_y - height
            width = 2 * width + 1
            height = 2 * height + 1
            rect_path = patches.Rectangle((bottom_left_x,bottom_left_y),width,height,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect_path)

        plt.show()

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
 

                
               
                
                
            
    
        