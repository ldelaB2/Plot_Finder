import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from functions import bindvec, flatten_mask_overlay
from PIL import Image
import os

class wavepad:
    def __init__(self, rawwavepad, expand_radi, path, output, disp_photo, disp = False):
        self.expand_radi = expand_radi
        self.imgsize = disp_photo.shape[0:2]
        self.path = path
        self.outputPath = output
        self.normaize_wavepad(rawwavepad)
        self.build_image(disp)
        self.filter_wavepad(disp_photo, disp)

    def normaize_wavepad(self, rawwavepad):
        tmp = np.array(rawwavepad)
        tmp = tmp.reshape(tmp.shape[0],2)
        tmp[:,0] = 1 - bindvec(tmp[:,0])
        tmp[:,1] = 1 - bindvec(tmp[:,1])
        self.wavepad = tmp

    def filter_wavepad(self, disp_photo, disp):
        _, self.row_binary = cv.threshold(self.row_wavepad, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)
        _, self.range_binary = cv.threshold(self.range_wavepad, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)

        row_filtered_disp = flatten_mask_overlay(disp_photo, self.row_binary, .5)
        range_filtered_disp = flatten_mask_overlay(disp_photo, self.range_binary, .5)

        #Saving Output
        tmp = Image.fromarray(row_filtered_disp)
        name = 'RowThresholdDisp.jpg'
        tmp.save(os.path.join(self.outputPath, name))

        tmp = Image.fromarray(range_filtered_disp)
        name = 'RangeThresholdDisp.jpg'
        tmp.save(os.path.join(self.outputPath, name))

        # Display output
        if disp:
            plt.imshow(row_filtered_disp)
            plt.show()
            plt.close()
            plt.imshow(range_filtered_disp)
            plt.show()
            plt.close()

    def build_image(self, disp):
        self.row_wavepad = np.zeros(self.imgsize)
        self.range_wavepad = np.zeros(self.imgsize)

        for e in range(self.wavepad.shape[0]):
            center = self.path.path[e]
            expand_radi = self.expand_radi
            rowstrt = center[1] - expand_radi[0]
            rowstp = center[1] + expand_radi[0] + 1
            colstrt = center[0] - expand_radi[1]
            colstp = center[0] + expand_radi[1] + 1

            self.row_wavepad[rowstrt:rowstp,colstrt:colstp] = self.wavepad[e,0]
            self.range_wavepad[rowstrt:rowstp,colstrt:colstp] = self.wavepad[e, 1]

        # Save the output
        self.row_wavepad = (self.row_wavepad * 255).astype(np.uint8)
        row_img = Image.fromarray(self.row_wavepad, mode = "L")
        name = 'Raw_Row_Wave.jpg'
        row_img.save(os.path.join(self.outputPath, name))

        self.range_wavepad  = (self.range_wavepad * 255).astype(np.uint8)
        range_img = Image.fromarray(self.range_wavepad, mode = "L")
        name = 'Range_Row_Wave.jpg'
        range_img.save(os.path.join(self.outputPath,name))
        print("Saved Wavepad QC")

        if disp:
            plt.imshow(row_output, cmap = 'grey')
            plt.show()
            plt.imshow(range_output, cmap = 'grey')
            plt.show()


