import numpy as np
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from matplotlib import pyplot as plt

from functions.display import disp_rectangles

class model():
    def __init__(self, param_dict):
        self.param_dict = param_dict
        self.model_shape = param_dict['model_shape']
        self.ncore = param_dict['ncore']


    def pca_filter(self, rect_list, threshold):
        model_list = []
        model_shape = [440, 109]

        for rect in rect_list:
            subImage = rect.create_sub_image()
            model_list.append(subImage.flatten())

        # Convert to numpy array
        models = np.array(model_list)
        # Compute the mean along the columns
        model_mean = np.mean(models, axis = 0)
        # Subtract the mean from the models
        models = models - model_mean
        # Compute the svd
        U, S, V = np.linalg.svd(models, full_matrices = False)

        # Using first 2 components
        proj_matrix = V.T[:, :2]

        # Project the models
        projected_models = models @ proj_matrix

        # Compute the distance from the origin
        distance_from_origin = np.linalg.norm(projected_models, axis = 1)
        # Compute the 75th percentile
        thres = np.percentile(distance_from_origin, threshold)

        # Threshold the models
        good_indx = distance_from_origin < thres

        # Compute the good models from the threshold
        good_models = models[good_indx]

        # Add the mean back to the models
        good_models = good_models + model_mean

        # Reshape the models
        good_models = good_models.reshape((-1, model_shape[0], model_shape[1]))

        return good_models

    def compute_base_img(self, rect_list):
        model_shape = self.param_dict['model_shape']
        x_radi = self.param_dict['x_radi']
        y_radi = self.param_dict['y_radi']
        img_shape = rect_list[0].img.shape

        # Compute the base image
        points = []
        for rect in rect_list:
            x = rect.center_x
            y = rect.center_y
            points.append([x,y])

        points = np.array(points)

        # Compute the bounds
        min_x = points[:,0].min() - x_radi - model_shape[1] // 2
        max_x = points[:,0].max() + x_radi + model_shape[1] // 2
        min_y = points[:,1].min() - y_radi - model_shape[0] // 2
        max_y = points[:,1].max() + y_radi + model_shape[0] // 2

        # Check the bounds
        min_x = max(0, min_x)
        max_x = min(img_shape[1], max_x)
        min_y = max(0, min_y)
        max_y = min(img_shape[0], max_y)

        # Compute the base image
        base_img = rect_list[0].img[min_y:max_y, min_x:max_x]
        base_img = base_img.astype(np.float32)

        x_bounds = [min_x, max_x]
        y_bounds = [min_y, max_y]

        return base_img, x_bounds, y_bounds

    def compute_feature_dict(self, good_models):
        feature_dict = {}
        lock = Lock()
        with ThreadPoolExecutor(max_workers = self.ncore) as executor:
            futures = [executor.submit(extract_features, (good_models[e,:,:], lock, feature_dict, e)) for e in range(good_models.shape[0])]

            # Wait for all the futures to finish
            for future in futures:
                future.result()
        
        return feature_dict 

    def compute_template_image(self, mean_model, base_img):
        results = cv.matchTemplate(base_img, mean_model, cv.TM_CCOEFF_NORMED)
        results = - results

        return results
    
    def compute_feature_center(self, feature_dict, rect_list, threshold):
        with ThreadPoolExecutor(max_workers = self.ncore) as executor:
            futures = [executor.submit(compute_feature_center, (feature_dict, rect, threshold)) for rect in rect_list]

            # Wait for all the futures to finish
            for future in futures:
                future.result()

    def compute_template_center(self, template_img, rect_list, x_bounds, y_bounds):
        for rect in rect_list:
            rect.compute_template_center(template_img, x_bounds, y_bounds)

    def move_rectangles(self, rect_list):
        for rect in rect_list:
            rect.compute_new_center()

    def sparce_optimize(self, rect_list, epoch):
        x_radi = self.param_dict['x_radi']
        y_radi = self.param_dict['y_radi']
        feature_threshold = .75

        # Compute the base image
        base_img, x_bounds, y_bounds = self.compute_base_img(rect_list)

        # Compute the bounds for each rectangle
        for rect in rect_list:
            rect.optimization_pre_process(x_radi, y_radi)

        for k in range(epoch):
            # PCA Filter the models
            good_models = self.pca_filter(rect_list, 75)

            # Compute the mean model and feature dictionary
            mean_model = np.mean(good_models, axis = 0).astype(np.float32)
            #feature_dict = self.compute_feature_dict(good_models)

            # Compute the template image
            template_img = self.compute_template_image(mean_model, base_img)

            # Compute the template image center
            self.compute_template_center(template_img, rect_list, x_bounds, y_bounds)
            # Compute the feature points center
            #self.compute_feature_center(feature_dict, rect_list, feature_threshold)

            # Move the rectangles
            self.move_rectangles(rect_list)
            print(f"Finished Epoch: {k+1}")


# Parallel Functions

def compute_template_center(args):
    rect, template_img, x_offset, y_offset = args
    rect.compute_template_center(template_img, x_offset, y_offset)
    return


def compute_feature_center(args):
    feature_dict, rect, threshold = args
    rect.compute_feature_center(feature_dict, threshold)
    return
            
def extract_features(args):
    img, lock, results_dict, key = args

    img = img.astype(np.uint8)

    orb = cv.ORB_create()
    kp, des = orb.detectAndCompute(img, None)

    with lock:
        results_dict[key] = {"keypoints": kp, "descriptors": des}
    
    return


def compute_temp_scores(args):
    model, base_img, results, lock = args
    result = cv.matchTemplate(base_img, model, cv.TM_CCOEFF_NORMED)
    result = - result

    with lock:
        results += result

    return

            