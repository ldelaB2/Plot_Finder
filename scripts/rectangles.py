import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift
from scipy.optimize import dual_annealing, Bounds, minimize
from PIL import Image

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


        
def create_unit_square(width, height):
    # Creating the unit square
    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    X, Y = np.meshgrid(x, y)
    unit_sqr = np.column_stack((X.ravel(), Y.ravel(), np.ones_like(X.ravel())))

    return unit_sqr 

def create_affine_frame(center_x, center_y, theta, width, height):
    width = (width / 2).astype(int)
    height = (height / 2).astype(int)
    theta = np.radians(theta)

    # Translation Matrix
    t_mat = np.zeros((3, 3))
    t_mat[0, 0], t_mat[1, 1], t_mat[2, 2] = 1, 1, 1
    t_mat[0, 2], t_mat[1, 2] = center_x, center_y

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

def compute_points(center_x, center_y, theta, width, height, unit_sqr, img_shape):
    affine_mat = create_affine_frame(center_x, center_y, theta, width, height)
    rotated_points = np.dot(affine_mat, unit_sqr.T).T
    rotated_points = rotated_points[:,:2].astype(int)

   
    # Checking to make sure points are within the image
    img_height, img_width = img_shape[:2]
    valid_y = (rotated_points[:, 1] >= 0) & (rotated_points[:, 1] < img_height)
    valid_x = (rotated_points[:, 0] >= 0) & (rotated_points[:, 0] < img_width)
    invalid_points = (~(valid_x & valid_y))
    rotated_points[invalid_points, :] = [0,0]

    return rotated_points

def extract_rectangle(center_x, center_y, theta, width, height, unit_sqr, img):
    points = compute_points(center_x, center_y, theta, width, height, unit_sqr, img.shape)

    if len(img.shape) > 2:
        extracted_img = img[points[:, 1], points[:, 0], :]
        extracted_img = np.reshape(extracted_img, (height, width, img.shape[2]))
    else:
        extracted_img = img[points[:, 1], points[:, 0]]
        extracted_img = np.reshape(extracted_img, (height, width))
        
    return extracted_img
             