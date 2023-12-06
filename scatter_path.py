import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from sub_image import sub_image

def compute_phase2_fun(args):
    FreqFilterWidth, row_mask, range_mask, num_pixles, center, image, boxradius = args
    raw_wavepad = np.zeros(2)
    subI = sub_image(image, boxradius, center)
    raw_wavepad[0] = subI.phase2(FreqFilterWidth, 0, row_mask, num_pixles)
    raw_wavepad[1] = subI.phase2(FreqFilterWidth, 1, range_mask, num_pixles)
    return(raw_wavepad)

class scatter_path:
    def __init__(self, image, boxradius, skip):
        self.image = image
        self.boxradius = boxradius
        self.skip = skip
        self.buildpath()

    def buildpath(self):
        image_dim = self.image.shape
        str1 = 1 + self.boxradius[0]
        stp1 = image_dim[0] - self.boxradius[0]
        str2 = 1 + self.boxradius[1]
        stp2 = image_dim[1] - self.boxradius[1]

        y = np.arange(str1, stp1, self.skip[0])
        x = np.arange(str2, stp2, self.skip[1])

        X, Y = np.meshgrid(x, y)
        self.path = np.column_stack((X.ravel(), Y.ravel()))
        self.num_points = self.path.shape[0]

    def disp_scatterpath(self):
        plt.imshow(self.image, cmap='gray')
        for point in self.path:
            plt.scatter(point[0], point[1], c='red', marker='*')
            plt.axis('on')
        plt.show()

    def compute_phase2(self, FreqFilterWidth, row_mask, range_mask, num_pixles, num_core = None):
        if num_core is None:
            num_cpu = multiprocessing.cpu_count()
        else:
            num_cpu = num_core
        print(f"Using {num_cpu} cores to process fine grid ")

        with multiprocessing.Pool(processes=num_cpu) as pool:
            results = pool.map(
                compute_phase2_fun, [(FreqFilterWidth, row_mask, range_mask, num_pixles, self.path[e], self.image,self.boxradius) for e in range(self.num_points)])

        return results

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
