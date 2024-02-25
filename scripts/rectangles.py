import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift
from scipy.optimize import dual_annealing, Bounds
from PIL import Image
from functions import create_unit_square, create_affine_frame, extract_rectangle

class rectangle:
    def __init__(self, rect):
        self.center_x = rect[1]
        self.center_y = rect[0]
        self.width = rect[2]
        self.height = rect[3]
        self.theta = rect[4]
        self.range = rect[5]
        self.row = rect[6]
        self.flagged = False
        self.ID = None

    def compute_histogram(self, img):
        sub_image = self.create_sub_image(img)
        # Compute the histogram for each channel
        red_histogram = np.histogram(sub_image[:,:,0], bins=256, range=(0, 256))
        green_histogram = np.histogram(sub_image[:,:,1], bins=256, range=(0, 256))
        blue_histogram = np.histogram(sub_image[:,:,2], bins=256, range=(0, 256))

        return red_histogram, green_histogram, blue_histogram
    
    def disp_histogram(self, img):
        red_histogram, green_histogram, blue_histogram = self.compute_histogram(img)
        # Plot the histogram
        fig, ax = plt.subplots(1)
        plt.plot(red_histogram[1][:-1], red_histogram[0], color='red')
        plt.plot(green_histogram[1][:-1], green_histogram[0], color='green')
        plt.plot(blue_histogram[1][:-1], blue_histogram[0], color='blue')
        plt.title("RGB Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.show()
        return fig, ax
    
    def compute_fft(self, img):
        sub_image = self.create_sub_image(img)
        sig = np.sum(sub_image, axis = 0)
        fsig = fftshift(fft(sig - np.mean(sig)))
        amp = abs(fsig)
    
        return amp
    
    def create_sub_image(self, img):
        unit_sqr = create_unit_square(self.width, self.height)
        sub_image = extract_rectangle(self.center_x, self.center_y, self.theta, self.width, self.height, unit_sqr, img)
        return sub_image

  
    def optomize_rectangle(self, img, model, x_radi, y_radi, theta_radi, miter):
        # Create objective function
        unit_sqr = create_unit_square(self.width, self.height)
        def objective_function(x):
            dX, dY, dTheta = x[0], x[1], x[2]
            dX = np.round(dX).astype(int)
            dY = np.round(dY).astype(int)
            dTheta = np.round(dTheta).astype(int)
            
            center_x = self.center_x + dX
            center_y = self.center_y + dY
            theta = self.theta + dTheta
            
            sub_img = extract_rectangle(center_x, center_y, theta, self.width, self.height, unit_sqr, img)
            dist = np.linalg.norm(sub_img - model)
            return dist

        # Optomize the rectangle
        bounds = Bounds([-x_radi, -y_radi, -theta_radi], [x_radi, y_radi, theta_radi])
        opt_solution = dual_annealing(objective_function, bounds, maxiter = miter)
        delta = opt_solution.x
        delta_x = delta[0].astype(int)
        delta_y = delta[1].astype(int)
        delta_theta = round(delta[2],1)

        # Update the rectangle
        self.center_x = self.center_x + delta_x
        self.center_y = self.center_y + delta_y
        self.theta = self.theta + delta_theta

    def save_rect(self, path, img):
        sub_img = self.create_sub_image(img)
        sub_img = sub_img.astype(np.uint8)
        sub_img = Image.fromarray(sub_img)
        sub_img.save(path)

    def compute_corner_points(self):
        points = np.array([[-1,1,1], [1,1,1], [1,-1,1], [-1,-1,1]])
        aff_mat = create_affine_frame(self.center_x, self.center_y, self.theta, self.width, self.height)
        corner_points = np.dot(aff_mat, points.T).T
        corner_points = corner_points[:,:2].astype(int)
        return corner_points
