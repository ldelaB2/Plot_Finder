from classes.plot_finder_job import plot_finder_job
import os, sys

def main(default_param_path, user_param_path, logger_path):
    job = plot_finder_job(default_param_path, user_param_path, logger_path)
    job.run()

    return
    
if __name__ == '__main__':
    param_path = sys.argv[1]
    #user_param_path = "/Users/willdelabretonne/Drone_Images/plot_finder_test_corn/params.json"
    default_param_path = os.path.join(os.path.dirname(__file__), "default_params.json")
    logger_path = os.path.join(os.path.dirname(param_path), "pf_log.log")

    main(default_param_path, param_path, logger_path)
    
    exit(0)