import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numpy.fft import fft, fftshift
from functions import  bindvec
from scipy.optimize import dual_annealing, minimize, Bounds
import multiprocessing as mp
from tqdm import tqdm

class rectangle:
    def __init__(self, rect):
        self.center = rect[0]
        self.theta = rect[1]
        self.range = rect[2][0]
        self.row = rect[2][1]
        self.points = None
        self.red_histogram = None
        self.green_histogram = None
        self.blue_histogram = None
        self.width = None
        self.height = None
        self.original_center = rect[0]
        self.original_theta = rect[1]
        self.old_center = None
        self.old_theta = None
        

class rectangle_list:
    def __init__(self, rect_list, mean_width, mean_height, img):
        self.rect_list = rect_list
        self.img = img
        self.mean_width = mean_width
        self.mean_height = mean_height
        self.create_unit_square()
        
        for e, rect in enumerate(self.rect_list):
            rect = rectangle(rect)
            rect.width = mean_width
            rect.height = mean_height
            self.rect_list[e] = rect


    def create_unit_square(self):
        self.unit_width = self.mean_width
        self.unit_height = self.mean_height
        # Creating the unit square
        y = np.linspace(-1, 1, self.unit_height)
        x = np.linspace(-1, 1, self.unit_width)
        X, Y = np.meshgrid(x, y)
        unit_sqr = np.column_stack((X.ravel(), Y.ravel(), np.ones_like(X.ravel())))
        self.unit_sqr = unit_sqr

    
    def create_affine_frame(self, rect):
        width = (self.unit_width / 2).astype(int)
        height = (self.unit_height / 2).astype(int)
        center = rect.center
        theta = rect.theta
        theta = np.radians(theta)

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
        affine_mat = self.create_affine_frame(rect)
        rotated_points = np.dot(affine_mat, self.unit_sqr.T).T
        rotated_points = rotated_points[:,:2].astype(int)

        # Checking to make sure points are within the image
        img_height, img_width = self.img.shape[:2]
        valid_y = (rotated_points[:, 1] >= 0) & (rotated_points[:, 1] < img_height)
        valid_x = (rotated_points[:, 0] >= 0) & (rotated_points[:, 0] < img_width)
        invalid_points = (~(valid_x & valid_y))
        rotated_points[invalid_points, :] = [0,0]

        rect.points = rotated_points
        

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
        self.compute_points(rect)

        rotated_points = rect.points
        img  = self.img
        z = img.shape
        if len(z) > 2:
            extracted_img = img[rotated_points[:, 1], rotated_points[:, 0], :]
            extracted_img = np.reshape(extracted_img, (self.unit_height, self.unit_width, img.shape[2]))
        else:
            extracted_img = img[rotated_points[:, 1], rotated_points[:, 0]]
            extracted_img = np.reshape(extracted_img, (self.unit_height, self.unit_width))
            
        return extracted_img
        

    def compute_model(self):
        model = np.zeros((self.mean_height, self.mean_width, 3))
        for rect in self.rect_list:
            sub_img = self.extract_rectangle(rect)
            model = model + sub_img

        model = (model / len(self.rect_list)).astype(int)
        self.model = model


    def disp_rectangles(self):
        fig, ax = plt.subplots(1)
        ax.imshow(self.img)

        for rect in self.rect_list:
            width = (self.mean_width / 2).astype(int)
            height = (self.mean_height / 2).astype(int)
            center_x, center_y, = rect.center[1], rect.center[0]
            bottom_left_x = center_x - width
            bottom_left_y = center_y - height
            rect_path = patches.Rectangle((bottom_left_x,bottom_left_y),self.mean_width,self.mean_height,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect_path)

        plt.show()
        return fig, ax


    def compute_fft_score(self):
        def compute_fft_score(img):
            sig = np.sum(img, axis = 0)
            fsig = fftshift(fft(sig - np.mean(sig)))
            amp = bindvec(abs(fsig))
            return amp

        scores = np.zeros((len(self.rect_list), self.mean_width))
        for e,rect in enumerate(self.rect_list):
            sub_img = self.extract_rectangle(rect)
            fsig = compute_fft_score(sub_img)
            scores[e,:] = fsig

        return scores
    
    def add_rectangles(self, new_rect_list):
        self.rect_list = self.rect_list + new_rect_list
 

    def optomize_rect_list(self, center_radi, theta_radi):
        for e in tqdm(range(len(self.rect_list)), desc = "Optomizing Rectangles"):
            rect = self.rect_list[e]
            optomize_rectangles((self.img, self.model, rect, center_radi, theta_radi, self.unit_sqr))


               
def optomize_rectangles(args):
    I, model, rect, center_radi, theta_radi, unit_sqr = args
    # Create objective function
    def objective_function(x):
        dX, dY, dTheta = x[0], x[1], x[2]

        center = rect.center.copy()
        theta = rect.theta
        center = center + np.array([dX, dY])
        theta = theta + dTheta
        sub_img = extract_rectangle(center, theta)
        dist = np.linalg.norm(sub_img - model)
        return dist

    # Extract rectangle from image
    def extract_rectangle(center, theta):
        points = compute_points(center, theta)
        sub_img = I[points[:, 1], points[:, 0], :]
        sub_img = np.reshape(sub_img, (rect.height, rect.width, I.shape[2]))
        return sub_img

    # Compute the points of rectangle
    def compute_points(center, theta):
        affine_mat = compute_affine_frame(center, theta)
        rotated_points = np.dot(affine_mat, unit_sqr.T).T
        rotated_points = rotated_points[:,:2].astype(int)

        # Checking to make sure points are within the image
        img_height, img_width = I.shape[:2]
        valid_y = (rotated_points[:, 1] >= 0) & (rotated_points[:, 1] < img_height)
        valid_x = (rotated_points[:, 0] >= 0) & (rotated_points[:, 0] < img_width)
        invalid_points = (~(valid_x & valid_y))
        rotated_points[invalid_points, :] = [0,0]
        return rotated_points
    
    def compute_affine_frame(center, theta):
        width = (rect.width / 2).astype(int)
        height = (rect.height / 2).astype(int)
        theta = np.radians(theta)

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
    
    center = rect.center.copy()
    theta = rect.theta
    bounds = Bounds([-center_radi, -center_radi, -theta_radi], [center_radi, center_radi, theta_radi])
    opt_solution = dual_annealing(objective_function, bounds, maxiter = 100)
    delta = opt_solution.x
    delta_center = delta[:2].astype(int)
    delta_theta = round(delta[2],1)
    rect.center = center + delta_center
    rect.theta = theta + delta_theta