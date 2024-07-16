
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
        self.flagged = False
        self.added = False
        self.ID = None
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
    

    def optimize_temp_match(self, model, param_dict):
        half_width = self.width // 2
        half_height = self.height // 2
        search_img_x_bound = [self.min_center_x - half_width, self.max_center_x + half_width]
        search_img_y_bound = [self.min_center_y - half_height, self.max_center_y + half_height]
        search_img = self.img[search_img_y_bound[0]:search_img_y_bound[1], search_img_x_bound[0]:search_img_x_bound[1]]
        
        if search_img.size == 0:
            return False

        model_mean = np.mean(model)
        model_std = np.std(model)
        search_img_mean = np.mean(search_img)
        search_img_std = np.std(search_img)
        if search_img_std == 0:
            return False
        
        search_img = (search_img - search_img_mean) / search_img_std
        model = (model - model_mean) / model_std
        search_img = search_img.astype(np.float32)
        model = model.astype(np.float32)
        results = cv.matchTemplate(search_img, model, cv.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(results)

        estimated_center_x = max_loc[0] + half_width
        current_center_x = search_img.shape[1] // 2
        estimated_center_y = max_loc[1] + half_height
        current_center_y = search_img.shape[0] // 2
        dx = estimated_center_x - current_center_x
        dy = estimated_center_y - current_center_y
        new_x = self.center_x + dx
        new_y = self.center_y + dy

        new_x = np.clip(new_x, self.min_center_x, self.max_center_x)
        new_y = np.clip(new_y, self.min_center_y, self.max_center_y)

        self.center_x = new_x
        self.center_y = new_y


    def disp_optimization(self, train_points, test_points, pred_model, model, loss):
        # Shrink the test_points
        indx = np.arange(0, test_points.shape[0], 5)
        test_points = test_points[indx]
        # Compute train obj_val
        train_obj = self.test_points(model, "XY", train_points, loss)
        # Compute all obj_val
        all_obj_val = self.test_points(model, "XY", test_points, loss)
        # Compute the predicted obj_val
        pred_obj_val = pred_model.predict(test_points)

        plt.close('all')
        # Plot the results
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(train_points[:,0], train_points[:,1], train_obj, c = 'b', marker = '.', label = 'Training Points')
        ax.scatter(test_points[:,0], test_points[:,1], all_obj_val, c = 'r', marker = '.', label = 'True Values')
        ax.scatter(test_points[:,0], test_points[:,1], pred_obj_val, c = 'g', marker = '.', label = 'Predicted Values')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Objective Function')
        plt.legend()
        plt.show()

    def optimize_theta(self, model, param_dict):
        loss = param_dict['optimization_loss']
        current_img = self.create_sub_image()
        current_score = compute_score(current_img, model, method = loss)
        thetas = []
        for theta in self.dtheta:
            new_img = self.move_rectangle(0, 0, theta)
            tmp_score = compute_score(new_img, model, method = loss)
            thetas.append(tmp_score)

        fopt = np.min(thetas)
        xopt = self.dtheta[np.argmin(thetas)]
        if fopt < current_score:
            self.theta += xopt
            update_flag = True
        else:
            update_flag = False
        
        return update_flag
    
    def optimize_HW(self, model, param_dict):
        loss = param_dict['optimization_loss']
        num_points = param_dict['ntest_HW']

        # Compute the current score
        current_score = compute_score(self.create_sub_image(), model, method = loss)

        # Create the meshgrid
        H, W = np.meshgrid(self.dheight, self.dwidth)
        all_points = np.column_stack((H.ravel(), W.ravel()))
        train_indx = np.random.choice(all_points.shape[0], num_points, replace = False)
        train_points = all_points[train_indx]
        all_points = np.delete(all_points, train_indx, axis = 0)

        # Fit the model
        pred_model, x_train_opt, f_train_opt = self.fit_model(model, "HW", train_points, loss, method = "RF", degree = 1)

        # Predict the objective function
        all_obj = pred_model.predict(all_points)
        # Sort the points
        sorted_indx = np.argsort(all_obj)
        sorted_indx = sorted_indx[:num_points]
        best_points = all_points[sorted_indx]

        # Test the best points
        best_obj_val = self.test_points(model, "HW", best_points, loss)
        best_x_opt = best_points[np.argmin(best_obj_val)]
        best_f_opt = np.min(best_obj_val)

        # Check if the optimization worked
        if best_f_opt < f_train_opt:
            fopt = best_f_opt
            xopt = best_x_opt
        else:
            fopt = f_train_opt
            xopt = x_train_opt

        # Update the rectangle if needed
        if fopt < current_score:
            self.width += xopt[1]
            self.height += xopt[0]
            self.recompute_unit_sqr = True
            update_flag = True
        else:
            update_flag = False


        return update_flag


    def optimize_XY(self, model, param_dict, display = False):
        loss = param_dict['optimization_loss']
        num_points = param_dict['ntest_XY']
        optimization_flag = param_dict['preform_XY_optimization']

        # Compute the current score
        current_score = compute_score(self.create_sub_image(), model, method = loss)

        # Compute the objective function
        X, Y = np.meshgrid(self.dx, self.dy)
        all_points = np.column_stack((X.ravel(), Y.ravel()))
        train_indx = np.random.choice(all_points.shape[0], num_points, replace = False)
        train_points = all_points[train_indx]
        all_points = np.delete(all_points, train_indx, axis = 0)

        if optimization_flag:
            # Fit the model
            pred_model, x_train_opt, f_train_opt = self.fit_model(model, "XY", train_points, loss, method = "RF", degree = 2)
            # Predict the objective function
            all_obj = pred_model.predict(all_points)

            # Find the best points
            sorted_indx = np.argsort(all_obj)
            sorted_indx = sorted_indx[:num_points]
            best_points = all_points[sorted_indx]
            
            # Test the best points
            best_obj_val = self.test_points(model, "XY", best_points, loss)
            best_x_opt = best_points[np.argmin(best_obj_val)]
            best_f_opt = np.min(best_obj_val)

            # Check if the optimization worked
            if best_f_opt < f_train_opt:
                fopt = best_f_opt
                xopt = best_x_opt
            else:
                fopt = f_train_opt
                xopt = x_train_opt

            if display:
                self.disp_optimization(train_points, all_points, pred_model, model, loss)
        else:
            obj_val = self.test_points(model, "XY", train_points, loss)
            fopt = np.min(obj_val)
            xopt = train_points[np.argmin(obj_val)]

        # Update the rectangle if needed
        if fopt < current_score:
            self.center_x += xopt[0]
            self.center_y += xopt[1]
            update_flag = True
        else:
            update_flag = False

        

        return update_flag

    def test_points(self, model, phase, points, loss):
        obj_val = []
        if phase == "XY":
            for point in points:
                new_img = self.move_rectangle(point[0], point[1], 0)
                tmp_score = compute_score(new_img, model, method = loss)
                obj_val.append(tmp_score)
        elif phase == "HW":
            for point in points:
                new_img = self.shrink_rectangle(point[1], point[0])
                tmp_score = compute_score(new_img, model, method = loss)
                obj_val.append(tmp_score)
        elif phase == "Theta":
            for point in points:
                new_img = self.move_rectangle(0, 0, point)
                tmp_score = compute_score(new_img, model, method = loss)
                obj_val.append(tmp_score)
        else:
            print("Invalid phase for fitting model")
            return
        
        return obj_val

    def fit_model(self, model, phase, train_points, loss, method = "RF", degree = 2):
        obj_val = self.test_points(model, phase, train_points, loss)

        train_points = np.array(train_points)
        if method == "RF":
            pred_model = make_pipeline(PolynomialFeatures(degree = degree), RandomForestRegressor())
        elif method == "LR":
            pred_model = make_pipeline(PolynomialFeatures(degree = degree), LinearRegression())
        else:
            print("Invalid method for fitting model")
            return
        
        pred_model.fit(train_points, obj_val)
        f_train_opt = np.min(obj_val)
        x_train_opt = train_points[np.argmin(obj_val)]

        return pred_model, x_train_opt, f_train_opt


    