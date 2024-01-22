import os, tracemalloc, time
from ortho_photo import ortho_photo
from functions import create_input_args

def main():
    # Starting to measure memory and time
    tracemalloc.start()
    start_time = time.time()

    # --------------------------Processing Ortho Photo ------------------------------------
    # Local
    raw_path = '/Users/willdelabretonne/Documents/Code/Python/OOP_Plot_Finder/test_imgs'
    #Docker
    #raw_path = sys.argv[1]  
    
    # Creating input arguments
    num_cores, input_path, output_path = create_input_args(raw_path)
    print(f"Using {num_cores} cores to process image")

    # Box radius is the size of sub images; 0 = height, 1 = width
    boxradius = (800, 500)
    
    # Create ortho photo classes
    print(f"Reading Images from: \n{input_path} \nSaving Output to: \n{output_path}")
    ortho_photos = [ortho_photo(input_path, output_path, image) for image in os.listdir(input_path)]

    # Loop through all photos to process
    for current_photo in ortho_photos:
        current_photo = ortho_photos[6]
        print(f"Starting to process photo {current_photo.name}")

        # Read in the ortho photo and create gray scale photo (needs work)
        current_photo.read_inphoto()

        # --------------------------Processing Spare Grid -----------------------------------
        # @Param sparse_skip: Step size for sparse grid
        sparse_skip = (100,100)
        # @Param FreqFilterWidth: Controls how many frequencies we let in when searching
        FreqFilterWidth = 1
        # @Param row_sig_remove: How many frequencies around the center to set to 0
        row_sig_remove = 5
        # @Param num_sig_returned: How many frequencies to include in the mask
        num_sig_returned = 2

        # Building Sparse Path then Compute Phase 1 and build frequency mask
        current_photo.build_scatter_path(boxradius, sparse_skip, expand_radi = None, disp = False)
        current_photo.phase1(FreqFilterWidth, num_sig_returned, row_sig_remove, disp = False)
        print("Finished processing sparse grid")

        # --------------------------Processing Fine Grid ------------------------------------
        #@Param expand_radi: How many pixels to return for each subI 0 = row, 1 = column
        expand_radi = [5,5]
        #@Param FreqFilterWidth_FineGrid: Controls how many frequencies we let in when searching Fine grid
        FreqFilterWidth = 1
        #@Param wave_pixel_expand: Controls how many positions in the wave are measured to find pixel value
        wave_pixel_expand = 0

        #Generate fine grid and process to create rawwavepad
        current_photo.build_scatter_path(boxradius, None, expand_radi, disp = False)
        current_photo.phase2(FreqFilterWidth, wave_pixel_expand, ncore = num_cores)
        print("Finished processing fine grid")

        # --------------------------Processing Wave Pad ------------------------------------
        # Creating wavepad object
        current_photo.build_wavepad(disp = False)
        # Finding training plots
        poly_degree_range = 3
        poly_degree_row = 1
         # Finding edge plots missed by FFT
        nrows = 96
        nranges = 6

        current_photo.find_plots(ncore = num_cores, poly_degree_range = poly_degree_range, poly_degree_col = poly_degree_row, nrange = nranges, nrow = nrows)
        
        # Optomizing plot locations
        miter = 5
        center_radi = 20
        theta_radi = 5
        current_photo.optomize_plots(miter, center_radi, theta_radi)

        # --------------------------Processing Plot Extraction ------------------------------
        # Extracting plots and creating geo shape file
        current_photo.extract_plots()
        current_photo.create_shapefile()

        print(f"""Finished Processing {current_photo.name}
              Thanks for using PLot Finder! Keep on Keeping on - Squid Billy Willy""")

        #Reporting memory and time usage
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        end_time = time.time()
        elapsed_time = end_time - start_time
       
        print(f"Current memory usage: {current_mem / (1024 ** 3):.2f} GB")
        print(f"Peak memory usage: {peak_mem / (1024 ** 3):.2f} GB")
        print(f"Time taken: {elapsed_time / 60:.2f} minutes")

    tracemalloc.stop()
    


if __name__ == '__main__':
    main()
