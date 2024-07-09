
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import dual_annealing, Bounds
from deap import base, creator, tools, algorithms
from pyswarm import pso
from functions.image_processing import create_unit_square, extract_rectangle
from functions.rectangle import five_2_four_rect, compute_score
from functions.general import minimize_quadratic
import random
import cv2 as cv
from functions.optimization_models import simulated_annealing
from scipy.optimize import minimize
from functions.general import bindvec

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
        self.initial_opt = False
        self.final_opt = False
        self.ID = None
        self.unit_sqr = None
        self.neighbors = None

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
        method = param_dict['method']
        
        if method == 'PSO':
            update_flag = self.optomize_rectangle_pso(model, param_dict)
        elif method == 'GA':
            update_flag = self.optomize_rectangle_ga(model, param_dict)
        elif method == 'SA':
            update_flag = self.optomize_rectangle_sa(model, param_dict)
        elif method == 'quadratic':
            update_flag = self.optomize_rectangle_quadratic(model, param_dict)
        else:
            print("Optimization method not recognized")

        return update_flag
    
    def optomize_rectangle_quadratic(self, model, param_dict):
        loss = param_dict['optimization_loss']
        test_points = param_dict['test_points']

        # Coompute the current objective function
        current_score = compute_score(self.create_sub_image(), model, method = loss)
        
        # Compute the objective function
        obj_val = []
        for point in test_points:
            new_img = self.move_rectangle(point[0], point[1], 0)
            new_img = bindvec(new_img)
            tmp_score = compute_score(new_img, model, method = loss)
            obj_val.append(tmp_score)

        # Fit the multivariate quadratic
        #test_points = np.array(test_points)
        #X = np.c_[test_points[:,0]**2, test_points[:,1]**2, test_points[:,0]*test_points[:,1], test_points[:,0], test_points[:,1], np.ones(test_points.shape[0])]
        #coeffs = np.linalg.lstsq(X, obj_val, rcond=None)[0]
        
        fmin = np.min(obj_val)
        xopt = test_points[np.argmin(obj_val)]

        if fmin < current_score:
            self.center_x += xopt[0]
            self.center_y += xopt[1]
            update_flag = True
        else:
            update_flag = False
        
        return update_flag


         

    def optomize_rectangle_pso(self, model, param_dict):
        # Pull the params
        x_radi = param_dict['x_radi']
        y_radi = param_dict['y_radi']
        theta_radi = param_dict['theta_radi']
        loss = param_dict['optimization_loss']
        swarm_size = param_dict['swarm_size']
        mxiter = param_dict['maxiter']

        # Define the objective function
        def objective_function(x):
            dX, dY, dTheta = x
        
            test_img = self.move_rectangle(dX, dY, dTheta)
            test_img = bindvec(test_img)
            dist = compute_score(test_img, model, method = loss)

            return dist
        
        # Define the bounds
        lb = [-x_radi, -y_radi, -theta_radi]
        ub = [x_radi, y_radi, theta_radi]

        # Optomize the rectangle using PSO
        xopt, fopt = pso(objective_function, lb, ub, swarmsize=swarm_size, maxiter=mxiter)

        # Check to make sure we are improving the model
        initial_fitness = objective_function([0,0,0])
        if initial_fitness < fopt:
            update_flag = False
            return update_flag
        else:
            delta_x = np.round(xopt[0]).astype(int)
            delta_y = np.round(xopt[1]).astype(int)
            delta_theta = np.round(xopt[2]).astype(int)

            self.center_x = self.center_x + delta_x
            self.center_y = self.center_y + delta_y
            self.theta = self.theta + delta_theta

            update_flag = True
            return update_flag
    
        

    def optomize_rectangle_ga(self, model, param_dict):
        x_radi = param_dict['x_radi']
        y_radi = param_dict['y_radi']
        theta_radi = param_dict['theta_radi']
        loss = param_dict['optimization_loss']
        num_pops = param_dict['num_pops']
        num_gens = param_dict['num_gens']
        mutation_prob = param_dict['mutation_prob']
        crossover_prob = param_dict['crossover_prob']

        def custom_mutate(individual):
            bounds = [(-x_radi, x_radi), (-y_radi, y_radi), (-theta_radi, theta_radi)]

            for i in range(len(individual)):
                if random.random() < 0.1:
                    individual[i] = random.randint(*bounds[i])
            return individual,
        
        def objective_function(x):
            dX, dY, dTheta = x
        
            test_img = self.move_rectangle(dX, dY, dTheta)
            test_img = bindvec(test_img)
            dist = compute_score(test_img, model, method = loss)
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

        # Preform the genetic algorithm
        pop = toolbox.population(n=num_pops)
        hof = tools.HallOfFame(1)
        algorithms.eaSimple(pop, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, ngen=num_gens, stats=None, halloffame=hof, verbose=False)
        xopt = hof[0]

        # Check to make sure we are improving the model
        initial_fitness = objective_function([0,0,0])
        fopt = objective_function(xopt)

        if initial_fitness < fopt:
            update_flag = False
            return update_flag
        else:
            delta_x = np.round(xopt[0]).astype(int)
            delta_y = np.round(xopt[1]).astype(int)
            delta_theta = np.round(xopt[2]).astype(int)

            self.center_x = self.center_x + delta_x
            self.center_y = self.center_y + delta_y
            self.theta = self.theta + delta_theta

            update_flag = True
            return update_flag
    
    def optomize_rectangle_sa(self, model, param_dict):
        # Pull the params
        x_radi = param_dict['x_radi']
        y_radi = param_dict['y_radi']
        theta_radi = param_dict['theta_radi']
        loss = param_dict['optimization_loss']
        mxiter = param_dict['maxiter']

        # Define the objective function
        def objective_function(x):
            dX, dY, dTheta = x
        
            test_img = self.move_rectangle(dX, dY, dTheta)
            test_img = bindvec(test_img)
            dist = compute_score(test_img, model, method = loss)

            return dist
        
        # Define the bounds
        bounds = Bounds([-x_radi, -y_radi, -theta_radi], [x_radi, y_radi, theta_radi])
        #bounds = [[-x_radi, -y_radi, -theta_radi], [x_radi, y_radi, theta_radi]]

        initial_temp = 5000
        min_temp = 1
        cooling_rate = 0.99
        step_size = 3
        max_iterations = 300

        #xopt, fopt = simulated_annealing(objective_function, bounds, initial_temperature = initial_temp, cooling_rate = cooling_rate, min_temperature = min_temp, max_iterations = max_iterations, step_size = step_size)


        mxiter = 1000
        # Optomize the rectangle using simulated annealing
        opt_solution = dual_annealing(objective_function, bounds, maxiter = mxiter)
        xopt = opt_solution.x

        # Check to make sure we are improving the model
        initial_fitness = objective_function([0,0,0])
        fopt = opt_solution.fun

        if initial_fitness < fopt:
            update_flag = False
            return update_flag
        else:
            delta_x = np.round(xopt[0]).astype(int)
            delta_y = np.round(xopt[1]).astype(int)
            delta_theta = np.round(xopt[2]).astype(int)

            self.center_x = self.center_x + delta_x
            self.center_y = self.center_y + delta_y
            self.theta = self.theta + delta_theta

            update_flag = True
            return update_flag
        


