import numpy as np
import cv2 as cv

from functions.general import bindvec


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
    if angle < 0:
        angle += 180
    rotation = np.round(angle - 90, 3)

    logger.info(f"Model Rotation: {rotation} degrees")

    center_point = (int(model_size[1] // 2), int(model_size[0] // 2))

    rotation_matrix = cv.getRotationMatrix2D(center_point, rotation, 1)

    rotation_matrix[0, 2] += hort_shift
    rotation_matrix[1, 2] += vert_shift

    translated_model = cv.warpAffine(initial_model, rotation_matrix, (model_size[1], model_size[0]))

    return translated_model

def compute_model_final(model_size, rect_list, logger):
    logger.info(f"Computing Final Model with {len(rect_list)} rectangles")

    final_model = np.zeros(model_size)
    for rect in rect_list:
        sub_img = rect.create_sub_image()
        if sub_img.shape != model_size:
            sub_img = cv.resize(sub_img, model_size[::-1])

        final_model += sub_img

    final_model = final_model / len(rect_list)
    final_model = np.round(255 * bindvec(final_model)).astype(np.uint8)

    return final_model


def compute_template_image(model, base_img):
    results = cv.matchTemplate(base_img, model, cv.TM_CCOEFF_NORMED)
    results = -results

    return results

def compute_template_score(rect_list, template_img, x_radi, y_radi):
    rect_list = set_radi(rect_list, x_radi, y_radi)
    scores = []
    for rect in rect_list:
        rect.optimize_xy(template_img)
        scores.append(rect.score)

    score = np.mean(scores)

    return score

def set_radi(rect_list, x_radi, y_radi, t_radi = None, h_radi = None, w_radi = None):
    for rect in rect_list:
        rect.compute_radi(x_radi, y_radi, t_radi, h_radi, w_radi)

    return rect_list

def compute_score(model, img, method):
    if img.shape != model.shape:
        img = cv.resize(img, model.shape[::-1])

    if method == "L2":
        score = np.linalg.norm(model - img)

    elif method == "CCOEFF":
        result = cv.matchTemplate(img, model, cv.TM_CCOEFF_NORMED)
        score = -result[0][0]

    return score












