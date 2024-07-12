import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from functions.general import bindvec


def build_rect_list(polygon_list, img):
    print("Building Rectangles")

def compute_model(rect_list):
    if len(rect_list[0].img.shape) == 2:
        model = np.zeros((rect_list[0].height, rect_list[0].width))
    else:
        model = np.zeros((rect_list[0].height, rect_list[0].width, rect_list[0].img.shape[2]))
    
    for rect in rect_list:
        sub_img = rect.create_sub_image()
        model += sub_img

    model = bindvec(model / len(rect_list))
    
    return model

def compute_score(img, model, method = "L2"):
    if method == "cosine":
        img_vec = img.flatten()
        img_norm = np.linalg.norm(img_vec)
        model_vec = model.flatten()
        model_norm = np.linalg.norm(model_vec)

        if img_norm == 0:
            print("Zero norm image")
            return np.inf
        else:
            cosine_similarity = np.dot(model_vec, img_vec) / (model_norm * img_norm)
            return cosine_similarity

    elif method == "L2":
        score = np.linalg.norm(img - model, 2)
        return score
    
    elif method == "L1":
        score = np.linalg.norm(img - model, 1)
        return score

    else:
        print("Invalid method for computing score")
        return

def compute_score_list(rect_list, model, method):
    scores = []
    for rect in rect_list:
        subI = rect.create_sub_image()
        subI = bindvec(subI)
        tmp_score = compute_score(subI, model, method)
        scores.append(tmp_score)

    final_score = np.median(scores)

    return final_score

def shrink_rect(rect, model, opt_param):
    width_shrink = opt_param['width_shrink']
    height_shrink = opt_param['height_shrink']
    loss = opt_param['optimization_loss']

    current_img = rect.create_sub_image()
    current_score = compute_score(bindvec(current_img), model, method = loss)
    current_height = rect.height
    current_width = rect.width

    # Shrink the width
    widths = np.arange(-width_shrink, 0, 1)
    w_scores = []
    for width in widths:
        new_img = current_img[:, abs(width):width:]
        new_img = cv.resize(new_img, (current_width, current_height))
        new_img = bindvec(new_img)
        tmp_score = compute_score(new_img, model, method = loss)
        w_scores.append(tmp_score)

    fopt = np.min(w_scores)
    w_opt = widths[np.argmin(w_scores)]
    if fopt < current_score:
        rect.width = rect.width + w_opt
        w_update = True
    else:
        w_update = False

    # Shrink the height
    heights = np.arange(-height_shrink, 0, 1)
    h_scores = []
    for height in heights:
        new_img = current_img[abs(height):height:,:]
        new_img = cv.resize(new_img, (current_width, current_height))
        new_img = bindvec(new_img)
        tmp_score = compute_score(new_img, model, method = loss)
        h_scores.append(tmp_score)

    fopt = np.min(h_scores)
    h_opt = heights[np.argmin(h_scores)]
    if fopt < current_score:
        rect.height = rect.height + h_opt
        h_update = True
    else:
        h_update = False

    return w_update, h_update








