import numpy as np
import matplotlib.pyplot as plt

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













