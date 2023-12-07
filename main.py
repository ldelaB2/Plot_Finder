import os, functions, tracemalloc, time, sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ortho_photo import ortho_photo

if __name__ == '__main__':
    # Starting to measure memory and time
    tracemalloc.start()
    start_time = time.time()

    # --------------------------Processing Ortho Photo ------------------------------------
    # Input and output directories (local)
    input_path = '/Users/willdelabretonne/Documents/PycharmProjects/OOP_Plot_Finder/input'
    output_path = '/Users/willdelabretonne/Documents/PycharmProjects/OOP_Plot_Finder/output'
    num_cores = 10
    # Docker t
    #input_path = sys.argv[1]
    #output_path = sys.argv[2]
    #num_cores = int(sys.argv[3])

    # Box radius is the size of sub images; 0 = height, 1 = width
    boxradius = (800, 500)

    # Create ortho photo classes
    print(f"Reading Images from: \n{input_path} \nSaving Output to: \n{output_path}")
    ortho_photos = [ortho_photo(input_path, output_path, image) for image in os.listdir(input_path)]

    for current_photo in ortho_photos:
        # Loop through all photos to process
        #current_photo = ortho_photos[6]
        print(f"Starting to process photo {current_photo.name}")
        # *-*-*-*-*-*-*-*-*-*-*-* Action Phase *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        # Read in the ortho photo and create gray scale photo (needs work)
        current_photo.read_inphoto()

        # --------------------------Processing Spare Grid -----------------------------------
        # @Param sparse_skip: Step size for sparse grid
        sparse_skip = (100,100)
        # @Param FreqFilterWidth: Controls how many frequencies we let in when searching sparse grid
        FreqFilterWidth_SparseGrid = 1
        # @Param vert_sig_remove: How many frequencies around the center to set to 0
        vert_sig_remove = 5
        # @Param num_sig_returned: How many frequencies to include in the mask
        num_sig_returned = 2
        # *-*-*-*-*-*-*-*-*-*-*-* Action Phase *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        # Building Sparse Path then Compute Phase 1 and build frequency mask
        current_photo.build_scatter_path(boxradius, sparse_skip, disp = False)
        current_photo.compute_phase1(FreqFilterWidth_SparseGrid, num_sig_returned, vert_sig_remove, disp = False)
        print("Finished processing sparse grid generating fine grid")

        # --------------------------Processing Fine Grid ------------------------------------
        #@Param expand_radi: How many pixels to return for each subI 0 = row, 1 = column
        expand_radi = [5,5]
        #@Param FreqFilterWidth_FineGrid: Controls how many frequencies we let in when searching Fine grid
        FreqFilterWidth_FineGrid = 1
        #@Param wave_pixel_expand: Controls how many positions in the wave are measured to find pixel value
        wave_pixel_expand = 0
        # *-*-*-*-*-*-*-*-*-*-*-* Action Phase *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        #Generate fine grid and process to create rawwavepad
        current_photo.build_scatter_path(boxradius, 0, expand_radi, disp = False)
        current_photo.compute_phase2(FreqFilterWidth_FineGrid, wave_pixel_expand, ncore = num_cores)
        print("Finished processing fine grid generating wave pad")

        # --------------------------Processing Wave Pad ------------------------------------
        #Creating wavepad object
        current_photo.build_wavepad()


    #Reporting memory and time usage
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    end_time = time.time()
    elapsed_time = end_time - start_time
    tracemalloc.stop()
    print(f"Current memory usage: {current_mem / (1024 ** 3):.2f} GB")
    print(f"Peak memory usage: {peak_mem / (1024 ** 3):.2f} GB")
    print(f"Time taken: {elapsed_time / 60:.2f} minutes")