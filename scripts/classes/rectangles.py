
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from scipy.optimize import dual_annealing, Bounds
from PIL import Image

from deap import base, creator, tools, algorithms
from pyswarm import pso
from operator import itemgetter

from functions.image_processing import create_unit_square, extract_rectangle
from functions.rectangle import five_2_four_rect
import random
from tqdm import tqdm

class rect_list:
    def __init__(self, rect_list, img, build_rectangles = True):
        self.img = img
        self.rect_list = rect_list
        if build_rectangles:
            self.build_rectangles()
        else:
            self.width = rect_list[0].width
            self.height = rect_list[0].height

    def build_rectangles(self):
        self.width = np.mean(np.array(list(map(itemgetter(2), self.rect_list)))).astype(int)
        self.height = np.mean(np.array(list(map(itemgetter(3), self.rect_list)))).astype(int)

        for idx, rect in enumerate(self.rect_list):
            rect[2] = self.width
            rect[3] = self.height
            self.rect_list[idx] = rectangle(rect)

    def compute_score(self):
        if self.model is None:
            self.build_model()
        score = 0
        for rect in self.rect_list:
            sub_image = rect.create_sub_image(self.img)
            score += np.linalg.norm(sub_image - self.model)
        
        return score

    def build_model(self):
        if len(self.img.shape) == 2:
            model = np.zeros((self.height, self.width))
        else:
            model = np.zeros((self.height, self.width, self.img.shape[2]))

        for rect in self.rect_list:
            sub_image = rect.create_sub_image(self.img)
            model += sub_image

        model = (model / len(self.rect_list)).astype(np.uint8)
        self.model = model

    def add_rectangles(self, rect_list):
        for rect in rect_list:
            self.rect_list.append(rect)

    def remove_rectangles(self, rect_list):
        remove_range = [rect.range for rect in rect_list]
        remove_row = [rect.row for rect in rect_list]

        current_range = [rect.range for rect in self.rect_list]
        current_row = [rect.row for rect in self.rect_list]

        remove_indx = np.where(np.isin(current_range, remove_range) & np.isin(current_row, remove_row))[0]
        self.rect_list = [self.rect_list[indx] for indx in range(len(self.rect_list)) if indx not in remove_indx]

    def optimize_rectangles(self, param_dict):
        num_updated = 0
        for k in tqdm(range(len(self.rect_list)), desc = "Optimizing Rectangles"):
            opt_flag = self.rect_list[k].optomize_rectangle(self.img, self.model, param_dict)
            num_updated += opt_flag
        
        print(f"Improved {num_updated}/{len(self.rect_list)} Rectangles")


    def disp_rectangles(self):
        output_img = np.copy(self.img)
        width = (self.width / 2).astype(int)
        
        for rect in self.rect_list:
            points = rect.compute_corner_points()

            if rect.flagged:
                output_img = cv.polylines(output_img, [points], True, (0, 255, 0), 10)
            else:
                output_img = cv.polylines(output_img, [points], True, (255, 0, 0), 10)

            if rect.ID is not None:
                font = cv.FONT_HERSHEY_SIMPLEX
                scale = 1.5
                color = (255,255,255)
                position = (rect.center_x - width, rect.center_y)
                thickness = 5
                txt = str(rect.ID)
                output_img = cv.putText(output_img, txt, position, font, scale, color, thickness, cv.LINE_AA)
        
        output_img = Image.fromarray(output_img)

        return output_img


class rectangle:
    def __init__(self, rect):
        self.center_y = rect[0]
        self.center_x = rect[1]
        self.width = rect[2]
        self.height = rect[3]
        self.theta = rect[4]
        self.range = rect[5]
        self.row = rect[6]
        self.flagged = False
        self.ID = None
        self.unit_sqr = None

    def compute_histogram(self, img):
        sub_image = self.create_sub_image(img)
        # Compute the histogram for each channel
        red_histogram = np.histogram(sub_image[:,:,0], bins=256, range=(0, 256))
        green_histogram = np.histogram(sub_image[:,:,1], bins=256, range=(0, 256))
        blue_histogram = np.histogram(sub_image[:,:,2], bins=256, range=(0, 256))

        return red_histogram, green_histogram, blue_histogram
    
    def disp_histogram(self, img):
        red_histogram, green_histogram, blue_histogram = self.compute_histogram(img)
        # Plot the histogram
        fig, ax = plt.subplots(1)
        plt.plot(red_histogram[1][:-1], red_histogram[0], color='red')
        plt.plot(green_histogram[1][:-1], green_histogram[0], color='green')
        plt.plot(blue_histogram[1][:-1], blue_histogram[0], color='blue')
        plt.title("RGB Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.show()
        return fig, ax
    
    def create_sub_image(self, img):
        unit_sqr = create_unit_square(self.width, self.height)
        sub_image = extract_rectangle(self.center_x, self.center_y, self.theta, self.width, self.height, unit_sqr, img)
        return sub_image
    
    def compute_corner_points(self):
        points = (self.center_x, self.center_y, self.width, self.height, self.theta)
        corner_points = five_2_four_rect(points)
        return corner_points

  
    def optomize_rectangle(self, img, model, param_dict):
        x_radi = param_dict['x_radi']
        y_radi = param_dict['y_radi']
        theta_radi = param_dict['theta_radi']
        method = param_dict['method']

        # Create objective function
        unit_sqr = create_unit_square(self.width, self.height)

        def objective_function(x):
            dX, dY, dTheta = x
        
            center_x = self.center_x + dX
            center_y = self.center_y + dY
            theta = self.theta + dTheta
            
            sub_img = extract_rectangle(center_x, center_y, theta, self.width, self.height, unit_sqr, img)
            dist = np.linalg.norm(sub_img - model)
            return dist
    
        
        if method == 'PSO':
            # Pull the parameters from the param_list
            swarm_size = param_dict['swarm_size']
            mxiter = param_dict['maxiter']

            lb = [-x_radi, -y_radi, -theta_radi]
            ub = [x_radi, y_radi, theta_radi]
            xopt, fopt = pso(objective_function, lb, ub, swarmsize=swarm_size, maxiter=mxiter)


        elif method == 'GA':
            def custom_mutate(individual):
                bounds = [(-x_radi, x_radi), (-y_radi, y_radi), (-theta_radi, theta_radi)]

                for i in range(len(individual)):
                    if random.random() < 0.1:
                        individual[i] = random.randint(*bounds[i])
                return individual,
        
            def objective_function(x):
                dX, dY, dTheta = x
            
                center_x = self.center_x + dX
                center_y = self.center_y + dY
                theta = self.theta + dTheta
                
                sub_img = extract_rectangle(center_x, center_y, theta, self.width, self.height, unit_sqr, img)
                dist = np.linalg.norm(sub_img - model)
                return dist,
        
            # Define the individual and population functions
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)

            # Define the individual and population functions
            toolbox = base.Toolbox()
            toolbox.register("attr_dx", random.randint, -x_radi, x_radi)  # bounds for dx
            toolbox.register("attr_dy", random.randint, -y_radi, y_radi)  # bounds for dy
            toolbox.register("attr_dtheta", random.randint, -theta_radi, theta_radi)  # bounds for dtheta

            toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_dx, toolbox.attr_dy, toolbox.attr_dtheta), n=1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            # Define the objective function
            toolbox.register("evaluate", objective_function)

            # Define the genetic operators
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", custom_mutate)
            toolbox.register("select", tools.selTournament, tournsize=3)

            # Create the population
            num_pops = param_dict['num_pops']
            num_gens = param_dict['num_gens']
            mutation_prob = param_dict['mutation_prob']
            crossover_prob = param_dict['crossover_prob']

            # Preform the genetic algorithm
            pop = toolbox.population(n=num_pops)
            hof = tools.HallOfFame(1)
            algorithms.eaSimple(pop, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, ngen=num_gens, stats=None, halloffame=hof, verbose=False)
            xopt = hof[0]


        elif method == 'SA':
            # Pull the parameters from the param_list
            mxiter = param_dict['maxiter']

            # Optomize the rectangle using simulated annealing
            bounds = Bounds([-x_radi, -y_radi, -theta_radi], [x_radi, y_radi, theta_radi])
            opt_solution = dual_annealing(objective_function, bounds, maxiter = mxiter)
            xopt = opt_solution.x


        # Check to make sure we are improving the model
        initial_fitness = objective_function([0,0,0])
        fopt = objective_function(xopt)

        if fopt < initial_fitness:
            delta_x = np.round(xopt[0]).astype(int)
            delta_y = np.round(xopt[1]).astype(int)
            delta_theta = np.round(xopt[2]).astype(int)

            self.center_x = self.center_x + delta_x
            self.center_y = self.center_y + delta_y
            self.theta = self.theta + delta_theta

            cnt = 1
        else:
            cnt = 0

        return cnt

         




