
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import dual_annealing, Bounds
from deap import base, creator, tools, algorithms
from pyswarm import pso
from functions.image_processing import create_unit_square, extract_rectangle
from functions.rectangle import five_2_four_rect
from functions.display import disp_flow
import random
import cv2 as cv

class rectangle:
    def __init__(self, rect):
        self.center_y = rect[0]
        self.center_x = rect[1]
        self.width = rect[2]
        self.height = rect[3]
        self.theta = rect[4]

        if len(rect) == 7:
            self.range = rect[5]
            self.row = rect[6]
        else:
            self.range = None
            self.row = None

        self.img = None
        self.model = None
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
    
    def create_sub_image(self):
        sub_image = extract_rectangle(self.center_x, self.center_y, self.theta, self.width, self.height, self.unit_sqr, self.img)
        return sub_image
    
    def compute_corner_points(self):
        points = (self.center_x, self.center_y, self.width, self.height, self.theta)
        corner_points = five_2_four_rect(points)
        return corner_points
  
    def move_rectangle(self, dX, dY, dT):
        center_x = self.center_x + dX
        center_y = self.center_y + dY
        theta = self.theta + dT
        
        new_img = extract_rectangle(center_x, center_y, theta, self.width, self.height, self.unit_sqr, self.img)
        
        return new_img



    def optomize_rectangle(self, model, param_dict):
        x_radi = param_dict['x_radi']
        y_radi = param_dict['y_radi']
        theta_radi = param_dict['theta_radi']
        method = param_dict['method']

        num_points = 10
        x_test = np.round(np.linspace(-x_radi, x_radi, num_points)).astype(int)
        img_1 = self.create_sub_image()
        
        x_angle = []
        x_mag = []
        x_hort = []

        for x_val in x_test:
            img_2 = self.move_rectangle(x_val, 0, 0)
        
            # Set the parameters for the Farneback method
            pyr_scale = 0.5  # Scale between image pyramids
            levels = 7       # Number of pyramid levels
            winsize = 20     # Size of the window for averaging
            iterations = 7   # Number of iterations at each pyramid level
            poly_n = 7       # Size of the pixel neighborhood
            poly_sigma = 1.5 # Standard deviation of the Gaussian
            flags = 0        # Flags for the algorithm

            # Compute optical flow using Farneback method
            flow = cv.calcOpticalFlowFarneback(img_1, img_2, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)

            # Compute the average flow vector
            avg_flow = np.mean(flow, axis=(0, 1))
            avg_hort_flow = np.mean(flow[:,:,0])

            # Compute the magnitude and angle of the average flow vector
            magnitude = np.sqrt(avg_flow[0]**2 + avg_flow[1]**2)
            angle = np.arctan2(avg_flow[1], avg_flow[0]) * 180 / np.pi 

            x_angle.append(angle)
            x_mag.append(magnitude)
            x_hort.append(avg_hort_flow) 


        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the first image in the first subplot
        axs[0].imshow(img_1)
        axs[0].set_title('Image 1')
        axs[0].axis('off')  # Hide the axes

        # Plot the second image in the second subplot
        axs[1].imshow(img_1)
        axs[1].set_title('Image 2')
        axs[1].axis('off')  # Hide the axes

        # Display the plot
        plt.show()





        """

        # Create objective function
        unit_sqr = create_unit_square(self.width, self.height)

        def objective_function(x):
            dX, dY, dTheta = x
        
            center_x = self.center_x + dX
            center_y = self.center_y + dY
            theta = self.theta + dTheta
            
            sub_img = extract_rectangle(center_x, center_y, theta, self.width, self.height, self.unit_sqr, self.img)
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
            reanneal_int = 20

            # Optomize the rectangle using simulated annealing
            bounds = Bounds([-x_radi, -y_radi, -theta_radi], [x_radi, y_radi, theta_radi])
            opt_solution = dual_annealing(objective_function, bounds, maxiter = mxiter, reanneal_interval = reanneal_int)
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

        """

         




