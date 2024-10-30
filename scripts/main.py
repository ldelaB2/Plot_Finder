from classes.plot_finder_job import plot_finder_job
import sys

def main(param_path):
    job = plot_finder_job(param_path)
    job.run()

    return
    
if __name__ == '__main__':
    #param_path = sys.argv[1]
    param_path = "/Volumes/will/Drone_Images/Masters_Datasets/Finding_Plots/Potato/HARS_2024/Trial_1/6.7.24/plot_finder/final_params.json"
    main(param_path)
    
    exit(0)