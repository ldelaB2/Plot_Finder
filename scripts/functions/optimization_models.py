import numpy as np



# Simulated Annealing algorithm
def simulated_annealing(objective_function,
                        bounds,
                        initial_temperature = None,
                        cooling_rate = None,
                        min_temperature = None,
                        max_iterations = None,
                        step_size = None):
    
    def perturb_solution(solution):
        # Randomly perturb the solution
        proposed_solution = solution + np.random.randint(-step_size, step_size, size = solution.shape)
        proposed_solution = np.clip(proposed_solution, lower_bound, upper_bound)

        return proposed_solution
    
    # Set default values for parameters
    if initial_temperature is None:
        initial_temperature = 1000
    if cooling_rate is None:
        cooling_rate = 0.999
    if min_temperature is None:
        min_temperature = 1
    if max_iterations is None:
        max_iterations = 1000
    if step_size is None:
        step_size = 4
    
    # Initialize the bounds, current solution, and best solution
    lower_bound = np.array(bounds[0])
    upper_bound = np.array(bounds[1])

    current_solution = np.random.randint(lower_bound, upper_bound)
    current_objective = objective_function(current_solution)

    best_solution = current_solution
    best_objective = current_objective
    
    temperature = initial_temperature
    
    for iteration in range(max_iterations):
        if temperature <= min_temperature:
            #print("Minimum temperature reached")
            break
        
        new_solution = perturb_solution(current_solution)
        new_objective = objective_function(new_solution)
        
        # Calculate the acceptance probability
        acceptance_probability = np.exp((current_objective - new_objective) / temperature)
        
        # Accept or reject the new solution
        if new_objective < current_objective or np.random.rand() < acceptance_probability:
            current_solution = new_solution
            current_objective = new_objective
        
        # Update the best solution found
        if current_objective < best_objective:
            best_solution = current_solution
            best_objective = current_objective
        
        # Cool down the temperature
        temperature *= cooling_rate
        
        # Optional: Print the progress
        #print(f"Iteration {iteration+1}: Best Solution = {best_solution}, Best Objective = {best_objective}")
        #print(f"Temperature = {temperature}")
    
    return best_solution, best_objective