import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from functions.general import bindvec

from functions.display import disp_rectangles

class model():
    def __init__(self, model_size):
        self.model_size = model_size

    def compute_initial_model(self, rect_list, logger):
        logger.info(f"Computing Initial Model with {len(rect_list)} rectangles")
        initial_model = self.compute_mean_model(rect_list)

        _, threshold_model = cv.threshold(initial_model, 0, 1, cv.THRESH_OTSU)

        num_obj, labeled, stats, centroids = cv.connectedComponentsWithStats(threshold_model)

        correct_object = np.argmax(stats[1:, 4]) + 1
        model_center = centroids[correct_object]
        model_center = np.round(model_center).astype(int)

        hort_shift = self.model_size[1] // 2 - model_center[0]
        vert_shift = self.model_size[0] // 2 - model_center[1]

        logger.info(f"Horizontal Shift for Initial Model: {hort_shift}")
        logger.info(f"Vertical Shift for Initial Model: {vert_shift}")

        translation_matrix = np.float32([[1, 0, hort_shift], [0, 1, vert_shift]])

        translated_model = cv.warpAffine(initial_model, translation_matrix, (self.model_size[1], self.model_size[0]))

        return translated_model
    

    def compute_mean_model(self, rect_list):
        model = np.zeros(self.model_size)
        for rect in rect_list:
            sub_img = rect.create_sub_image()
            if sub_img.shape != self.model_size:
                sub_img = cv.resize(sub_img, self.model_size[::-1])

            model += sub_img

        model = model / len(rect_list)
        model = np.round(255 * bindvec(model)).astype(np.uint8)

        return model
    

def compute_template_image(model, base_img):
    results = cv.matchTemplate(base_img, model, cv.TM_CCOEFF_NORMED)
    results = -results

    return results
    
   

            