import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from functions.general import bindvec, geometric_median
import cv2 as cv
from functions.display import disp_rectangles


def optimize_rect_list_xy(rect_list):
    scores = []
    for rect in rect_list:
        rect.compute_template_score_image()
        scores.append(rect.score)
        rect.compute_template_position()

    score = np.mean(scores)

    return score










