import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from functions.general import bindvec
import cv2 as cv


def build_rect_list(polygon_list, img):
    print("Building Rectangles")

def compute_model(rect_list, model_shape, shift = False):
    model = np.zeros(model_shape)
   
    for rect in rect_list:
        sub_img = rect.create_sub_image()
        if sub_img.shape != model_shape:
            sub_img = cv.resize(sub_img, model_shape[::-1])

        sub_mean = np.mean(sub_img)
        sub_std = np.std(sub_img)
        if sub_std == 0:
            sub_img = np.zeros(model_shape)
        else:
            sub_img = (sub_img - sub_mean) / sub_std

        model += sub_img

    model = model / len(rect_list)
    model_mean = np.mean(model)
    model_std = np.std(model)

    if model_std == 0:
        print("Zero std model")
        return model
    else:
        model = ((model - model_mean) / model_std).astype(np.float32)

    if shift:
        # Threshold the model
        tmp = (bindvec(model) * 255).astype(np.uint8)
        _, binary_model = cv.threshold(tmp, 0, 1, cv.THRESH_OTSU)
        # Compute the moments
        M = cv.moments(binary_model)
        # Compute the center of mass
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = model_shape[1] // 2, model_shape[0] // 2

        dX = model_shape[1] // 2 - cX
        dY = model_shape[0] // 2 - cY

        # Compute the affine matrix
        affine_matrix = np.float32([[1, 0, dX], [0, 1, dY]])

        # Shift the model
        model = cv.warpAffine(model, affine_matrix, model_shape[::-1])


    return model

def compute_score(img, model, method = "L2"):
    if img.shape != model.shape:
        img = cv.resize(img, model.shape[::-1])

    img_mean = np.mean(img)
    img_std = np.std(img)
    if img_std == 0:
        return -np.inf
    else:
        img = (img - img_mean) / img_std
        img = img.astype(np.float32)

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

    final_score = np.mean(scores)

    return final_score










