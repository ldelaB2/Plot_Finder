import os, sys
from ortho_photo import ortho_photo
from functions import set_params

def main(param_path):
    # --------------------------Processing Ortho Photos ------------------------------------
    params = set_params(param_path) # set parameters
    print(f"Using {params['num_cores']} cores to process image")
    print(f"Reading Images from: \n{params['input_path']} \nSaving Output to: \n{params['output_path']}")

    current_photo = ortho_photo(params) # Create ortho photo object
    print(f"Starting to process photo {current_photo.name}")

    current_photo.read_inphoto() # Read in the ortho photo and prepare it for processing
    
    current_photo.phase1() # Compute phase 1
    print("Finished processing sparse grid")

    current_photo.phase2() # Compute phase 2
    
    if params["optomize_plots"] == True:
        current_photo.optomize_plots() # Optomize plots
    if params["save_plots"] == True:
        current_photo.save_plots() # Save plots
    if params["create_shapefile"] == True:
        current_photo.create_shapefile() # Create shapefile
    
    print(f"""Finished Processing {current_photo.name}
            Thanks for using PLot Finder! Keep on Keeping on - Squid Billy Willy""")

   
if __name__ == '__main__':
    param_path = sys.argv[1]
    #param_path = "/Users/willdelabretonne/Drone_Images/plot_finder_test_corn/params.json"
    main(param_path)