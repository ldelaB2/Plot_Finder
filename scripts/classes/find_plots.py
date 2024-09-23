from matplotlib import pyplot as plt
import numpy as np
import multiprocessing, os

from classes.sub_image import sub_image
from classes.wave_pad import wavepad
from functions.image_processing import build_path
from functions.general import create_shapefile


class find_plots():
    def __init__(self, plot_finder_job_params, loggers):
        self.params = plot_finder_job_params
        self.loggers = loggers
        self.phase_one() # Create the sparse grid and find the dominant frequencies
        self.phase_two() # Create the fine grid and find the wavepad
        self.phase_three() # Find the plots
    
    def phase_one(self):
        # Phase one calcualte the optimal signal
        # Pulling the params
        row_sig_remove = self.params["row_sig_remove"]
        sparse_skip_radi = 100
        num_sig_returned = self.params["num_sig_returned"]
        
        # Creating the sparse grid
        sparse_grid, num_points = build_path(self.params["img_ortho_shape"][:2], self.box_radi, sparse_skip_radi * 2 + 1)
        
        # Preallocate memory
        range_waves = np.zeros((num_points, (2 * self.box_radi[0])))
        row_waves = np.zeros((num_points, (2 * self.box_radi[1])))

        # Loop through sparse grid; returning the abs of Freq Wave
        for e in range(num_points):
            subI = sub_image(self.g_ortho, self.box_radi, sparse_grid[e])
            row_waves[e, :], range_waves[e, :] = subI.phase1(self.freq_filter_width)

        # Finding dominant frequency in row (column) direction
        row_sig = np.mean(row_waves, 0)
        # Finding dominant frequency in range (row) direction
        range_sig = np.mean(range_waves, 0)

        if row_sig_remove is not None:
            start = self.box_radi[1] - row_sig_remove
            end = self.box_radi[1] + row_sig_remove
            row_sig[start:end] = 0

        # Creating the masks
        self.row_mask = create_phase2_mask(row_sig, num_sig_returned)
        self.range_mask = create_phase2_mask(range_sig, num_sig_returned)

        print("Finished Processing Sparse Grid")
    
    def phase_two(self):
        # Pulling the params
        fine_skip_radi = self.pf_job.params["fine_grid_radi"]
        num_cores = self.pf_job.params["num_cores"]

        # Build the fine grid
        fine_grid, num_points = build_path(self.g_ortho.shape, self.box_radi, fine_skip_radi * 2 + 1)
        
        # Parallelize the computation of the wavepad
        with multiprocessing.Pool(processes=num_cores) as pool:
            rawwavepad = pool.map(
                compute_phase2_fun,
                [(self.freq_filter_width, self.row_mask, self.range_mask, fine_grid[e], self.g_ortho, self.box_radi, fine_skip_radi) for e in range(num_points)])

        # Preallocate memory
        row_wavepad = np.ones_like(self.g_ortho).astype(np.float64)
        range_wavepad = np.ones_like(self.g_ortho).astype(np.float64)

        # Loop through the rawwavepad and place the snips in the correct location
        for e in range(len(rawwavepad)):
            center = rawwavepad[e][2]
            col_min = center[0] - fine_skip_radi
            col_max = center[0] + fine_skip_radi + 1
            row_min = center[1] - fine_skip_radi
            row_max = center[1] + fine_skip_radi + 1

            row_snp = np.tile(rawwavepad[e][0], (fine_skip_radi * 2 + 1, 1))
            range_snp = np.tile(rawwavepad[e][1], (fine_skip_radi * 2 + 1, 1)).T

            row_wavepad[row_min:row_max, col_min:col_max] = row_snp
            range_wavepad[row_min:row_max, col_min:col_max] = range_snp

        # Invert the wavepad
        row_wavepad = 1 - row_wavepad
        range_wavepad = 1 - range_wavepad
        self.raw_row_wavepad = (row_wavepad * 255).astype(np.uint8)
        self.raw_range_wavepad = (range_wavepad * 255).astype(np.uint8)

        print("Finished Processing Fine Grid")
    
    def phase_three(self):
        # Pass off to wavepad to find fft rectangles
        fft_placement = wavepad(self.raw_range_wavepad, self.raw_row_wavepad, self.pf_job.params, self.g_ortho)
        fft_rect_list = fft_placement.final_rect_list
        
        shp_path = os.path.join(self.pf_job.output_paths['shape_dir'], f"{self.pf_job.params['img_name']}_fft.gpkg")
        create_shapefile(fft_rect_list, self.pf_job.meta_data, self.inverse_rotation, shp_path)

        # Update the params file
        self.pf_job.params['shapefile_path'] = shp_path

        print("Finished FFT Rectangle Placement")


def create_phase2_mask(signal, num_sig_returned):
        ssig = np.argsort(signal)[::-1]
        freq_index = ssig[:num_sig_returned]
        mask = np.zeros_like(signal)
        mask[freq_index] = 1
    
        return mask

def compute_phase2_fun(args):
    FreqFilterWidth, row_mask, range_mask, center, image, boxradius, expand_radi = args
    subI = sub_image(image, boxradius, center)
    row_snip = subI.phase2(FreqFilterWidth, 0, row_mask, expand_radi)
    range_snip = subI.phase2(FreqFilterWidth, 1, range_mask, expand_radi)

    return (row_snip, range_snip, center)

