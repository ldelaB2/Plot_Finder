import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from sub_image import sub_image



class scatter_path:
    def __init__(self, img_shape, boxradius, skip):
        self.img_shape = img_shape
        self.boxradius = boxradius
        self.skip = skip
        self.buildpath()

        

    

    def compute_phase1(self, FreqFilterWidth, disp = False):
        # Preallocate memory
        range_waves = np.zeros((self.num_points, (2 * self.boxradius[0])))
        row_waves = np.zeros((self.num_points, (2 * self.boxradius[1])))

        # Loop through sparse grid; returning the abs of Freq Wave
        for e in range(self.num_points):
            center = self.path[e]
            subI = sub_image(self.image, self.boxradius, center)
            row_waves[e, :], range_waves[e, :] = subI.phase1(FreqFilterWidth)

        # Finding dominant frequency in row (column) direction
        row_sig = np.mean(row_waves, 0)
        # Finding dominant frequency in range (row) direction
        range_sig = np.mean(range_waves, 0)

        if disp:
            fig, axes = plt.subplots(nrows=1, ncols=2)
            axes[0].plot(row_sig)
            axes[0].set_title('Avg Row Signal')
            axes[1].plot(range_sig)
            axes[1].set_title('Avg Range Signal')
            plt.tight_layout()
            plt.show()

        return row_sig, range_sig
