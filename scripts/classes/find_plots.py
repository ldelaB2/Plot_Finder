from matplotlib import pyplot as plt
import numpy as np
import multiprocessing, os
from multiprocessing import shared_memory
import time

from classes.sub_image import sub_image
from classes.wave_pad import wavepad
from functions.image_processing import build_path
from functions.general import create_shapefile
from functions.pre_processing import compute_signal, compute_skip



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

        with multiprocessing.Pool(num_cores) as pool:
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
        # Pass off to wavepad to find fft rectangles
        fft_placement = wavepad(self.raw_range_wavepad, self.raw_row_wavepad, self.pf_job.params, self.g_ortho)
        fft_rect_list = fft_placement.final_rect_list
        
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