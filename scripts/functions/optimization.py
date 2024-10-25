import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from functions.display import disp_rectangles
from functions.general import bindvec

def compute_model(model_size, rect_list, logger):
    logger.info(f"Computing Initial Model with {len(rect_list)} rectangles")
  
    initial_model = np.zeros(model_size)
    for rect in rect_list:
        sub_img = rect.create_sub_image()
        if sub_img.shape != model_size:
            sub_img = cv.resize(sub_img, model_size[::-1])

        model += sub_img

    model = model / len(rect_list)
    model = np.round(255 * bindvec(model)).astype(np.uint8)

    _, threshold_model = cv.threshold(initial_model, 0, 1, cv.THRESH_OTSU)

    num_obj, labeled, stats, centroids = cv.connectedComponentsWithStats(threshold_model)

    correct_object = np.argmax(stats[1:, 4]) + 1
    model_center = centroids[correct_object]
    model_center = np.round(model_center).astype(int)

    hort_shift = model_size[1] // 2 - model_center[0]
    vert_shift = model_size[0] // 2 - model_center[1]

    logger.info(f"Horizontal Shift for Initial Model: {hort_shift}")
    logger.info(f"Vertical Shift for Initial Model: {vert_shift}")

    translation_matrix = np.float32([[1, 0, hort_shift], [0, 1, vert_shift]])

    translated_model = cv.warpAffine(initial_model, translation_matrix, (model_size[1], model_size[0]))

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


def optimize_xy(rect_list, x_radi, y_radi, g_img, model):
    template_img = compute_template_image(model, g_img)
    for rect in rect_list:
        rect.compute_radi(x_radi, y_radi)
        rect.template_img = template_img
        rect.compute_template_score_image()
        rect.compute_template_position()

    return rect_list


def optimize_t(rect_list, t_radi, model):
    t_range = np.arange(-t_radi, t_radi + 1, 1)

    # Create the images
    images = []
    for rect in rect_list:
        images.append(rect.create_sub_image())

    # Create the models
    output = []
    for t in t_range:
        (height, width) = model.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv.getRotationMatrix2D(center, t, 1)
        rotated_model = cv.warpAffine(model, rotation_matrix, (width, height))
        
        output.append(match_template_t((rotated_model, images)))

    output = np.array(output).T
    min_indices = np.argmin(output, axis = 1)
    opt_theta = t_range[min_indices]

    # Update the rectangles
    for idx, rect in enumerate(rect_list):
        rect.theta += opt_theta[idx]

    return rect_list

def optimize_hw(rect_list, h_radi, w_radi, model):
    d_height = np.round(h_radi * model.shape[0]).astype(int)
    d_width = np.round(w_radi * model.shape[1]).astype(int)
    num_samples = 10

    height_range = np.linspace(-d_height, 0, num_samples)
    width_range = np.linspace(-d_width, 0, num_samples)

    height_range = np.round(height_range).astype(int)
    width_range = np.round(width_range).astype(int)

    H, W = np.meshgrid(height_range, width_range)
    samples = np.vstack([H.ravel(), W.ravel()]).T

    results = []
    for sample in samples:
        results.append(match_template_hw((model, sample, rect_list)))

    results = np.array(results).T
    best_shrink = samples[np.argmin(results, axis = 1)]

    for idx, rect in enumerate(rect_list):
        rect.width += best_shrink[idx][1]
        rect.height += best_shrink[idx][0]

    return rect_list


def match_template_hw(args):
    model, delta, rect_list = args
    output = []
    for rect in rect_list:
        img = rect.shrink_rectangle(delta[1], delta[0])
        img = cv.resize(img, model.shape[::-1])

        result = cv.matchTemplate(img, model, cv.TM_CCOEFF_NORMED)
        output.append(-result[0][0])

    return np.array(output)


def match_template_t(args):
    model, base_imgs = args
    output = []
    for img in base_imgs:
        if img.shape != model.shape:
            img = cv.resize(img, model.shape[::-1])
        
        results = cv.matchTemplate(img, model, cv.TM_CCOEFF_NORMED)
        output.append(-results[0][0])

    return np.array(output)

def match_t_worker(args):
    rect, t_range, model = args
    sub_image = rect.create_sub_image()










