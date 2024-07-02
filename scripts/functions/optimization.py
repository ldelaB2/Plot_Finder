from classes.rectangles import rectangle
from functions.rectangle import compute_score
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

def compute_model(rect_list, initial = False):
    # Create the model object
    if len(rect_list[0].img.shape) == 2:
        model = np.zeros((rect_list[0].height, rect_list[0].width))
    else:
        model = np.zeros((rect_list[0].height, rect_list[0].width, rect_list[0].img.shape[2]))

    if initial:
        # Preallocate memory
        distance = np.zeros((len(rect_list), len(rect_list)))

        # Compute the distance matrix (only lower triangle because symetric)
        for cnt1, rect1 in enumerate(rect_list):
            img1 = rect1.create_sub_image()
            for cnt2 in range(cnt1, len(rect_list)):
                rect2 = rect_list[cnt2]
                img2 = rect2.create_sub_image()
                tmp_score = compute_score(img1, img2, method = "SSIM")
                distance[cnt1, cnt2] = tmp_score
        
        # Filling in the other side of symetric matrix
        distance = distance + distance.T

        # Compute the sum and median
        distance_sum = np.sum(distance, axis = 1)
        distance_median = np.median(distance_sum)
        
        # Find rects below the median
        good_rect_indx = np.argwhere(distance_sum <= distance_median)

        # Compute the model
        for indx in good_rect_indx:
            rect = rect_list[indx[0]]
            subI = rect.create_sub_image()
            model += subI

        model = np.round((model / len(good_rect_indx))).astype(np.uint8)

    else:
        cnt = 0
        for rect in rect_list:
            if not rect.flagged:
                subI = rect.create_sub_image()
                model += subI
                cnt += 1
            else:
                pass

        model = np.round((model / cnt)).astype(np.uint8)
    
    print("Finished computing model")
    return model

def optimize_list(rect_list, model, opt_param_dict, txt = "Optimizing Rectangles"):
    # If using quadratic pre compute the test points for the optimization
    if opt_param_dict['method'] == 'quadratic':
        # Pull the parameters
        x_radi = opt_param_dict['x_radi']
        y_radi = opt_param_dict['y_radi']
        num_points = opt_param_dict['quadratic_num_points']

        # Create the test points
        x = np.round(np.linspace(-x_radi, x_radi, num_points)).astype(int)
        y = np.round(np.linspace(-y_radi, y_radi, num_points)).astype(int)

        # Remove the zeros
        x = x[x != 0]
        y = y[y != 0]

        # Only keep unique values
        x = np.unique(x)
        y = np.unique(y)

        # Push to the dictionary
        opt_param_dict['test_x'] = x
        opt_param_dict['test_y'] = y

    # Pull the number of epochs
    epoch = opt_param_dict['epoch']

    for e in range(epoch):
        print(f"Starting Epoch {e + 1}/{epoch}")
        results = []
        for rect in tqdm(rect_list, total = len(rect_list), desc = txt):
            tmp = rect.optomize_rectangle(model, opt_param_dict)
            results.append(tmp)

        num_updated = np.sum(results)
        print(f"Updated {num_updated} rectangles")

    return