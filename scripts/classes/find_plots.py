from matplotlib import pyplot as plt
import numpy as np
import multiprocessing as mp
import os
from multiprocessing import shared_memory
import time

from classes.sub_image import sub_image
from classes.wave_pad import wavepad
from classes.model import model, compute_template_image

from functions.image_processing import build_path
from functions.general import create_shapefile
from functions.pre_processing import compute_signal, compute_skip
from functions.wavepad import process_wavepad
from functions.display import dialate_skel, flatten_mask_overlay, disp_rectangles
from functions.rect_list import build_rect_list, set_range_row, set_id
from functions.rect_list_processing import add_rectangles, remove_rectangles, distance_optimize, double_check, setup_rect_list




class find_plots():
    def __init__(self, plot_finder_job_params, loggers):
        self.params = plot_finder_job_params
        self.loggers = loggers
        self.phase_one() # Create the sparse grid and find the dominant frequencies

    
    def phase_one(self):
        # Phase one calcualte the optimal signal
        logger = self.loggers.fft_processing
        row_signal, range_signal = compute_signal(self.params, logger)

        # Box size
        box_radi = self.params["box_radi"]
        # Create the phase 2 mask
        row_mask = np.zeros((box_radi[1] * 2))
        range_mask = np.zeros((box_radi[0] * 2))

        row_mask[row_signal] = 1
        range_mask[range_signal] = 1

        logger.info("Finished Phase 1")
        self.phase_two(row_mask, range_mask)
    
    def phase_two(self, row_mask, range_mask):
        # Pulling the params
        logger = self.loggers.fft_processing
        fine_skip_radi = self.params["fine_skip_radi"]
        fine_skip_num_images = self.params["fine_skip_num_images"]
        num_cores = self.params["num_cores"]
        img_size = self.params["img_ortho_shape"]
        box_radi = self.params["box_radi"]
        gray_ortho = self.params["gray_img"]
        freq_filter_width = self.params["freq_filter_width"]

        # Check fi fine skip radi is defined
        if fine_skip_radi is None:
            logger.info(f"Fine skip radi not defined calculating using {fine_skip_num_images} images")
            fine_skip_radi = compute_skip(img_size[:2], box_radi, fine_skip_num_images, logger)
            
            # Update the params
            self.params["fine_skip_radi"] = fine_skip_radi
        else:
            logger.info(f"Using fine skip radi of {fine_skip_radi}")

        # Build the fine grid
        fine_grid, num_points = build_path(img_size, box_radi, fine_skip_radi)

        # Let the people know whats up
        logger.info(f"Starting to process fine grid using {num_points} images and {num_cores} cores")
        
        # Preallocate memory
        # Create shared memory for row_wavepad and range_wavepad
        array_size = int(np.prod(gray_ortho.shape))
        shm_row = shared_memory.SharedMemory(create=True, size=array_size)
        shm_range = shared_memory.SharedMemory(create=True, size=array_size)
        shm_gray = shared_memory.SharedMemory(create=True, size=array_size)
        shared_gray_img = np.ndarray(gray_ortho.shape, dtype = np.uint8, buffer = shm_gray.buf)
        np.copyto(shared_gray_img, gray_ortho)

        # Compute the expand radi
        expand_radi = np.array(fine_skip_radi) // 2
       
        # Shared params
        fine_grid_params = [row_mask,
                            range_mask,
                            freq_filter_width,
                            box_radi,
                            expand_radi,
                            shm_row.name,
                            shm_range.name,
                            shm_gray.name,
                            gray_ortho.shape]
        
        # Record the start time
        start_time = time.time()

        with mp.Pool(num_cores) as pool:
            args = [(fine_grid_params, center) for center in fine_grid]
            pool.map(phase_two_worker, args)
            
        # Pull the range and row wavepad out of shared memory
        row_wavepad = np.ndarray(gray_ortho.shape, dtype=np.uint8, buffer=shm_row.buf)
        range_wavepad = np.ndarray(gray_ortho.shape, dtype=np.uint8, buffer=shm_range.buf)

        # Close the shared memory
        shm_row.unlink()
        shm_range.unlink()
        shm_gray.unlink()

        end_time = time.time()
        logger.info(f"Finished processing fine grid in {np.round(end_time - start_time,2)} seconds")

        self.phase_three(row_wavepad, range_wavepad)
    
    def phase_three(self, row_wavepad, range_wavepad):
        # pull the params
        logger = self.loggers.wavepad_processing
        poly_deg_range = self.params["poly_deg_range"]
        poly_deg_row = self.params["poly_deg_row"]
        min_obj_size_range = self.params["min_obj_size_range"]
        min_obj_size_row = self.params["min_obj_size_row"]
        closing_iterations = self.params["closing_iterations"]
        img = self.params["gray_img"]

        # Process the wavepad
        range_skel = process_wavepad(range_wavepad, poly_deg_range, "range", min_obj_size_range, closing_iterations, logger)
        row_skel = process_wavepad(row_wavepad, poly_deg_row, "row", min_obj_size_row, closing_iterations, logger)

        initial_rect_list, initial_range_cnt, initial_row_cnt, initial_width, initial_height = build_rect_list(range_skel, row_skel, img)

        # Remote initial range and row
        logger.info(f"Initial Range Count: {initial_range_cnt}")
        logger.info(f"Initial Row Count: {initial_row_cnt}")

        # Compute the initial model
        model_size = (initial_height, initial_width)
        initial_model = model(model_size).compute_initial_model(initial_rect_list, logger)

        user_models = {}
        user_models["initial_model"] = initial_model

        # Update the params
        self.params["fft_range_cnt"] = initial_range_cnt
        self.params["fft_row_cnt"] = initial_row_cnt
        self.params["models"] = user_models
        self.params["model_size"] = model_size

        # Phase 4
        self.phase_four(initial_rect_list)
        
    def phase_four(self, initial_rect_list):
        # Pull the params
        logger = self.loggers.find_plots
        num_ranges = self.params["number_ranges"]
        num_rows = self.params["number_rows"]
        fft_ranges = self.params["fft_range_cnt"]
        fft_rows = self.params["fft_row_cnt"]
        initial_model = self.params["models"]["initial_model"]
        model_shape = self.params["model_size"]
        x_radi = np.round(self.params["x_radi"] * model_shape[1]).astype(int)
        y_radi = np.round(self.params["y_radi"] * model_shape[0]).astype(int)

        # Create the template img 
        template_image = compute_template_image(initial_model, self.params["gray_img"])
        
        # Setup the rect list
        setup_rect_list(initial_rect_list, x_radi, y_radi, template_image, model_shape)

        # Compute the number of ranges and rows to add or remove
        delta_ranges = num_ranges - fft_ranges
        delta_rows = num_rows - fft_rows

        # Ranges
        if delta_ranges > 0:
            logger.info(f"Adding {delta_ranges} missing ranges")
            initial_rect_list = add_rectangles(initial_rect_list, "range", delta_ranges, logger)
        elif delta_ranges < 0:
            logger.info(f"Removing {abs(delta_ranges)} extra ranges")
            initial_rect_list = remove_rectangles(initial_rect_list, "range", abs(delta_ranges), logger)
        
        logger.info("Double checking ranges")
        initial_rect_list = double_check(initial_rect_list, "range", logger)

        # Rows
        if delta_rows > 0:
            logger.info(f"Adding {delta_rows} missing rows")
            initial_rect_list = add_rectangles(initial_rect_list, "row", delta_rows, logger)
        elif delta_rows < 0:
            logger.info(f"Removing {abs(delta_rows)} extra rows")
            initial_rect_list = remove_rectangles(initial_rect_list, "row", abs(delta_rows), logger)
        
        logger.info("Double checking rows")
        initial_rect_list = double_check(initial_rect_list, "row", logger)

        logger.info("Finished finding all ranges and rows")

        self.phase_five(initial_rect_list)

    def phase_five(self, initial_rect_list):
        # Pulling the params
        logger = self.loggers.find_plots
        label_start = self.params["label_start"]
        label_flow = self.params["label_flow"]
        output_dir = self.params["pf_output_directorys"]
        shp_directory = output_dir["shapefiles"]
        img_name = self.params["image_name"]
        neighbor_radi = self.params["neighbor_radi"]
        distance_weight = 1

        # Setting the range and row
        set_range_row(initial_rect_list)
        set_id(initial_rect_list, start = label_start, flow = label_flow)

        # Distance Optimize
        final_rect_list = distance_optimize(initial_rect_list, neighbor_radi, distance_weight, logger)

        
        # Setting the id
        
        print("Finished Adding Labels")

        # Create the shapefile
        shp_path = os.path.join(self.pf_job.output_paths['shape_dir'], f"{self.pf_job.params['img_name']}_fft.gpkg")
        create_shapefile(fft_rect_list, self.pf_job.meta_data, self.inverse_rotation, shp_path)

        # Update the params file
        self.pf_job.params['shapefile_path'] = shp_path

        print("Finished FFT Rectangle Placement")


def phase_two_worker(args):
    phase_two_params, center = args
    row_mask, range_mask, freq_filter_width, box_radi, expand_radi, shm_row_name, shm_range_name, shm_gray_name, gray_ortho_shape = phase_two_params
    
    # Get the shared memory addresses
    row_shared_mem  = shared_memory.SharedMemory(name=shm_row_name)
    range_shared_mem = shared_memory.SharedMemory(name=shm_range_name)
    gray_shared_mem = shared_memory.SharedMemory(name=shm_gray_name)
    
    # Get the shared memory arrays
    row_wavepad = np.ndarray(gray_ortho_shape, dtype=np.uint8, buffer=row_shared_mem.buf)
    range_wavepad = np.ndarray(gray_ortho_shape, dtype=np.uint8, buffer=range_shared_mem.buf)
    gray_ortho = np.ndarray(gray_ortho_shape, dtype=np.uint8, buffer=gray_shared_mem.buf)

    # Run phase 2 of sub image
    subI = sub_image(gray_ortho, box_radi, center)
    row_wave = subI.phase2(freq_filter_width, 0, row_mask, expand_radi[1])
    range_wave = subI.phase2(freq_filter_width, 1, range_mask, expand_radi[0])

    min_row = center[1] - expand_radi[0]
    max_row = center[1] + expand_radi[0] + 1

    min_col = center[0] - expand_radi[1]
    max_col = center[0] + expand_radi[1] + 1

    row_snp = np.tile(row_wave, (expand_radi[0] * 2 + 1, 1))
    range_snp = np.tile(range_wave, (expand_radi[1] * 2 + 1, 1)).T

    row_wavepad[min_row:max_row, min_col:max_col] = row_snp
    range_wavepad[min_row:max_row, min_col:max_col] = range_snp

    row_shared_mem.close()
    range_shared_mem.close()
    gray_shared_mem.close()
   
    return