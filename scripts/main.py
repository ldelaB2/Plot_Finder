from classes.plot_finder_job import plot_finder_job

def main(param_path):
    job = plot_finder_job(param_path)
    job.run()
    
if __name__ == '__main__':
    #param_path = sys.argv[1]
    param_path = "/Users/willdelabretonne/Drone_Images/plot_finder_test_corn/params.json"
    main(param_path)