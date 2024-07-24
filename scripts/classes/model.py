import numpy as np
from matplotlib import pyplot as plt
from functions.display import disp_rectangles
from functions.distance_optimize import geometric_median
import multiprocessing as mp

class model():
    def __init__(self, rect_list, param_dict):
        self.rect_list = rect_list
        self.param_dict = param_dict
        self.compute_initial_model()

    def compute_initial_model(self):
        # Pull the params
        epoch = self.param_dict['epoch']
        x_radi = self.param_dict['x_radi']
        y_radi = self.param_dict['y_radi']
        ncore = self.param_dict['ncore'] 

        for k in range(epoch):
            print(f"Building Model epoch {k+ 1}")

            model_list = self.build_models_from_list(self.rect_list)
            
            # Compute the template scores
            with mp.Pool(processes=ncore) as pool:
                results = pool.map(compute_temp_scores, [(rect, model_list, x_radi, y_radi) for rect in self.rect_list])

            for e, result in enumerate(results):
                reshaped = result.reshape(result.shape[0], -1)
                max_indices_flat = np.argmax(reshaped, axis = 1)
                max_indices = np.column_stack(np.unravel_index(max_indices_flat, result.shape[1:]))

                # Compute the geometric median
                geom_median = geometric_median(np.array(max_indices))
                geom_median = np.round(geom_median).astype(int)

                # Compute the change
                dx = geom_median[1] - x_radi
                dy = geom_median[0] - y_radi

                # Update the rect
                self.rect_list[e].center_x = self.rect_list[e].center_x + dx
                self.rect_list[e].center_y = self.rect_list[e].center_y + dy

        self.initial_model = self.build_models_from_list(self.rect_list)


    def build_models_from_list(self, rect_list):
        model_list = []
        for rect in rect_list:
            subImage = rect.create_sub_image()
            img_mean = np.mean(subImage)
            img_std = np.std(subImage)
            subImage = (subImage - img_mean) / img_std
            subImage = subImage.astype(np.float32)
            model_list.append(subImage)
        
        return model_list

def compute_temp_scores(args):
    rect, models, x_radi, y_radi = args
    results = []
    for model in models:
        results.append(rect.compute_template_score(model, x_radi, y_radi))
    
    results = np.array(results)
    return results

            