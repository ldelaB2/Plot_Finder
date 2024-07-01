from classes.rectangles import rectangle

import numpy as np
from functions.image_processing import create_unit_square
import multiprocessing as mp
from tqdm import tqdm

def build_rect_list(rect_list, img):
        output_list = []
        # Find the mean width and height
        mean_width = np.mean([rect[2] for rect in rect_list])
        mean_height = np.mean([rect[3] for rect in rect_list])

        # Round the mean width and height
        mean_width = np.round(mean_width).astype(int)
        mean_height = np.round(mean_height).astype(int)

        # Create the unit square
        unit_sqr = create_unit_square(mean_width, mean_height)

        # Create the rectangles
        for rect in rect_list:
            rect[2] = mean_width
            rect[3] = mean_height

            rect = rectangle(rect)
            rect.img = img
            rect.unit_sqr = unit_sqr
            output_list.append(rect)

        return output_list

def compute_model(rect_list):
    if len(rect_list[0].img.shape) == 2:
        model = np.zeros((rect_list[0].height, rect_list[0].width))
    else:
        model = np.zeros((rect_list[0].height, rect_list[0].width, rect_list[0].img.shape[2]))

    for rect in rect_list:
        subI = rect.create_sub_image()
        model += subI

    model = (model / len(rect_list)).astype(np.uint8)

    return model

def optimize_list(rect_list, model, opt_param_dict, txt = "Optimizing Rectangles"):
    results = []
    for rect in tqdm(rect_list, total = len(rect_list), desc = txt):
        tmp = rect.optomize_rectangle(model, opt_param_dict)
        results.append(tmp)

    num_updated = np.sum(results)
    print(f"Updated {num_updated} rectangles")

    return