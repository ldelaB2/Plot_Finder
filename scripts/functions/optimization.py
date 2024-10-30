import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from functions.display import disp_rectangles
from functions.general import bindvec
from multiprocessing import shared_memory

def compute_model(model_size, rect_list, logger):
    logger.info(f"Computing Initial Model with {len(rect_list)} rectangles")
  
    initial_model = np.zeros(model_size)
    for rect in rect_list:
        sub_img = rect.create_sub_image()
        if sub_img.shape != model_size:
            sub_img = cv.resize(sub_img, model_size[::-1])

        initial_model += sub_img

    initial_model = initial_model / len(rect_list)
    initial_model = np.round(255 * bindvec(initial_model)).astype(np.uint8)

    _, threshold_model = cv.threshold(initial_model, 0, 1, cv.THRESH_OTSU)

    num_obj, labeled, stats, centroids = cv.connectedComponentsWithStats(threshold_model)

    correct_object = np.argmax(stats[1:, 4]) + 1
    model_center = centroids[correct_object]
    moments = cv.moments((labeled == correct_object).astype(np.uint8))

    model_center = np.round(model_center).astype(int)

    hort_shift = model_size[1] // 2 - model_center[0]
    vert_shift = model_size[0] // 2 - model_center[1]

    logger.info(f"Model Horozontal shift: {hort_shift} pixels")
    logger.info(f"Model Vertical shift: {vert_shift} pixels")

    # Calculate orientation angle (rotation)
    angle = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02']) * (180 / np.pi)
    rotation = np.round(angle - 90, 3)

    logger.info(f"Model Rotation: {rotation} degrees")

    center_point = (int(model_size[1] // 2), int(model_size[0] // 2))

    rotation_matrix = cv.getRotationMatrix2D(center_point, rotation, 1)

    rotation_matrix[0, 2] += hort_shift
    rotation_matrix[1, 2] += vert_shift

    translated_model = cv.warpAffine(initial_model, rotation_matrix, (model_size[1], model_size[0]))

    return translated_model

def compute_template_image(model, base_img):
    results = cv.matchTemplate(base_img, model, cv.TM_CCOEFF_NORMED)
    results = -results

    return results

def optimize_rect_list_xy(rect_list):
    scores = []
    for rect in rect_list:
        rect.compute_template_score_image()
        scores.append(rect.score)
        rect.compute_template_position()

    score = np.median(scores)

    return score

def set_radi(rect_list, x_radi, y_radi, t_radi, h_radi, w_radi):
    for rect in rect_list:
        rect.compute_radi(x_radi, y_radi, t_radi, h_radi, w_radi)

    return rect_list

def total_optimization(args):
    rect_list, model, template_img_name, template_img_shape, temp_dtype = args
    temp_shared_mem = shared_memory.SharedMemory(name = template_img_name)
    template_img = np.ndarray(template_img_shape, dtype = temp_dtype, buffer = temp_shared_mem.buf)

    for rect in rect_list:
        # XY Optimization
        rect.optimize_xy(template_img)

        # T Optimization
        rect.optimize_t(model, method = "L2")

        # H Optimization
        rect.optimize_height_width(model, method = "L2")


    temp_shared_mem.close()

    return

def compute_score(model, img, method):
    if img.shape != model.shape:
        img = cv.resize(img, model.shape[::-1])

    if method == "L2":
        score = np.linalg.norm(model - img)

    elif method == "CCOEFF":
        result = cv.matchTemplate(img, model, cv.TM_CCOEFF_NORMED)
        score = -result[0][0]

    return score












