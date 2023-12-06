import os, functions
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from functions import bindvec
from scatter_path import scatter_path
from wave_pad import wavepad

class ortho_photo:
    def __init__(self, in_path, out_path, name):
        self.name = name[:-4]
        self.in_path = os.path.join(in_path, name)
        self.out_path = out_path
        self.create_output_dirs()

    def create_output_dirs(self):
        subdir = os.path.join(self.out_path, self.name)
        self.QC_path = os.path.join(subdir, "QC")
        self.plots_path = os.path.join(subdir, "plots")
        try:
            os.mkdir(subdir)
            os.mkdir(self.plots_path)
            os.mkdir(self.QC_path)
        except:
            pass

    def read_inphoto(self):
        self.rgb_ortho = cv.imread(self.in_path)
        self.create_g()

    def create_g(self):
        self.g_ortho = cv.cvtColor(self.rgb_ortho, cv.COLOR_BGR2LAB)[:,:,2]

    def build_scatter_path(self, boxradius, skip, expand_radi = None, disp = False):
        if expand_radi is not None:
            skip = ((1 + 2 * expand_radi[0]), (1 + 2 * expand_radi[1]))
            self.expand_radi = expand_radi

        self.boxradius = boxradius
        self.PointGrid = scatter_path(self.g_ortho, boxradius, skip)

        if disp:
            self.PointGrid.disp_scatterpath()

    def build_wavepad(self):
        self.row_wavepad = np.zeros(self.g_ortho.shape).astype(np.uint8)
        self.range_wavepad = np.zeros(self.g_ortho.shape).astype(np.uint8)

        for e in range(self.PointGrid.num_points):
            center = self.PointGrid.path[e]
            expand_radi = self.expand_radi
            rowstrt = center[1] - expand_radi[0]
            rowstp = center[1] + expand_radi[0] + 1
            colstrt = center[0] - expand_radi[1]
            colstp = center[0] + expand_radi[1] + 1

            self.row_wavepad[rowstrt:rowstp, colstrt:colstp] = self.rawwavepad[e, 0]
            self.range_wavepad[rowstrt:rowstp, colstrt:colstp] = self.rawwavepad[e, 1]

        # Saving the output for Quality Control
        name = 'Raw_Row_Wave.jpg'
        Image.fromarray(self.row_wavepad).save(os.path.join(self.QC_path, name))
        name = 'Range_Row_Wave.jpg'
        Image.fromarray(self.range_wavepad).save(os.path.join(self.QC_path, name))
        print("Saved Wavepad QC")

    def compute_phase1(self, FreqFilterWidth, num_sig_returned, vert_sig_remove, disp = False):
        #Find the signals
        row_sig, range_sig = self.PointGrid.compute_phase1(FreqFilterWidth, disp)

        # Creating the masks
        self.row_mask = self.create_phase2_mask(row_sig, num_sig_returned, self.boxradius[1], vert_sig_remove, disp)
        self.range_mask = self.create_phase2_mask(range_sig, num_sig_returned, disp)

    def compute_phase2(self, FreqFilterWidth, wave_pixel_expand, ncore = None):
        self.rawwavepad = self.PointGrid.compute_phase2(FreqFilterWidth, self.row_mask, self.range_mask, wave_pixel_expand, ncore)
        self.rawwavepad = np.array(self.rawwavepad).reshape(-1,2)
        self.rawwavepad[:,0] = 1 - bindvec(self.rawwavepad[:,0])
        self.rawwavepad[:,1] = 1 - bindvec(self.rawwavepad[:,1])
        self.rawwavepad = (self.rawwavepad * 255).astype(np.uint8)

    def create_phase2_mask(self, signal, numfreq, radius=None, supressor=None, disp=False):
        if supressor is not None:
            signal[(radius - supressor):(radius + supressor)] = 0
            ssig = np.argsort(signal)[::-1]
        else:
            ssig = np.argsort(signal)[::-1]

        freq_index = ssig[:numfreq]
        mask = np.zeros_like(signal)
        mask[freq_index] = 1

        if disp:
            plt.plot(mask)
            plt.show()
        return mask
