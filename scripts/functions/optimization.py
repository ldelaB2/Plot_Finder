import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from functions.general import bindvec


def build_rect_list(polygon_list, img):
    print("Building Rectangles")

def compute_model(rect_list, model_shape):
    model = np.zeros(model_shape)
   
    for rect in rect_list:
        sub_img = rect.create_sub_image()
        if sub_img.shape != model_shape:
            sub_img = cv.resize(sub_img, model_shape[::-1])
        model += sub_img

    model = model / len(rect_list)

    model_mean = np.mean(model)
    model_std = np.std(model)

    if model_std == 0:
        print("Zero std model")
        return model
    else:
        model = ((model - model_mean) / model_std).astype(np.float32)
    
    return model

def compute_score(img, model, method = "L2"):
    if img.shape != model.shape:
        img = cv.resize(img, model.shape[::-1])

    img_mean = np.mean(img)
    img_std = np.std(img)
    if img_std == 0:
        return np.inf
    else:
        img = (img - img_mean) / img_std

    if method == "cosine":
        img_vec = img.flatten()
        img_norm = np.linalg.norm(img_vec)
        model_vec = model.flatten()
        model_norm = np.linalg.norm(model_vec)

        if img_norm == 0:
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
    
    elif method == "NCC":
        ncc = cv.matchTemplate(img, model, cv.TM_CCORR_NORMED)[0]
        ncc = -ncc
        return ncc

    else:
        print("Invalid method for computing score")
        return

def compute_score_list(rect_list, model, method):
    scores = []
    for rect in rect_list:
        subI = rect.create_sub_image()
        tmp_score = compute_score(subI, model, method)
        scores.append(tmp_score)

    final_score = np.median(scores)

    return final_score










