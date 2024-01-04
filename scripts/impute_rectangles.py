import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from rectangles import rectangle

def impute_rectangles(train_rect, col_skel, range_skel, img, ncol, nrange):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20,20))
    dialated_col_skel = cv.dilate(col_skel, kernel)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (100,100))
    dialated_range_skel = cv.dilate(range_skel, kernel)

    found_col, labeled_col, _, _ = cv.connectedComponentsWithStats(dialated_col_skel)

