import numpy as np
import cv2 as cv
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
from functions import bindvec, findmaxpeak

class sub_image:
    def __init__(self, image, boxradius, center):
        self.original_image = image
        self.boxradius = boxradius
        self.center = center
        self.extract_image()
        self.original_image = None
        self.axis = 0

    def phase2(self,FreqFilterWidth, direction, mask, num_pixels):
        self.axis = direction
        self.computeFFT()
        self.filterFFT(mask, FreqFilterWidth)
        self.generateWave()
        self.convertWave2Spacial()
        self.calcPixelValue(num_pixels)
        return self.pixelval

    def phase1(self, FreqFilterWidth, DispOutput = False):
        #Processing rows
        self.axis = 0
        self.computeFFT()
        self.filterFFT(None, FreqFilterWidth)
        self.generateWave()
        row_wave = abs(self.FreqWave)

        # Pretty graphs
        if DispOutput:
            self.disp_subI()
            self.plotFFT()
            self.plotMask()
            self.plotFreqWave()

        #Processing ranges
        self.axis = 1
        self.computeFFT()
        self.filterFFT(None, FreqFilterWidth)
        self.generateWave()
        range_wave = abs(self.FreqWave)

        #Pretty graphs
        if DispOutput:
            self.disp_subI()
            self.plotFFT()
            self.plotMask()
            self.plotFreqWave()

        return row_wave, range_wave

    def computeFFT(self):
        # Dir(1) = rows    Dir(0) = columns
        # Computing fft
        sig = np.mean(self.image, axis = self.axis)
        fsig = fft(sig - np.mean(sig))
        fsig = fftshift(fsig)
        self.FFT_amp = bindvec(abs(fsig))
        self.FFT_phi = np.angle(fsig)

    def filterFFT(self, mask, FreqFilterWidth):
        #Finding max peak
        max_peak = findmaxpeak(self.FFT_amp, mask)
        # Create Mask
        mask = np.zeros_like(self.FFT_amp)
        mask[max_peak] = 1
        # Dilate Mask
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (FreqFilterWidth,FreqFilterWidth))
        self.mask = cv.dilate(mask, kernel, iterations = 1).flatten()

    def generateWave(self):
        #Computing Frequency Wave
        amp = self.FFT_amp * self.mask
        phi = self.FFT_phi * self.mask
        self.FreqWave = amp * np.exp(1j * phi)

    def convertWave2Spacial(self):
        # Compute Spacial Wave
        self.SpacialWave = np.real(ifft(fftshift(self.FreqWave)))

    def calcPixelValue(self, num_pixels, disp = False):
        proj = np.zeros_like(self.mask)
        tindex = 1 - self.axis
        if num_pixels == 0:
            proj[self.boxradius[tindex]] = 1
        else:
            proj[(self.boxradius[tindex] - num_pixels):(self.boxradius[tindex] + num_pixels + 1)] = 1
        self.pixelval = np.float32((np.dot(self.SpacialWave, proj)) / np.sum(proj == 1))

        if disp:
            print(np.sum(proj == 1))
            plt.plot(proj)
            plt.show()

    def plotSpacialWave(self):
        fig, axes = plt.subplots(nrows=1, ncols=1)
        axes.plot(self.SpacialWave)
        axes.set_title('Spacial Wave')
        plt.show()

    def plotFreqWave(self):
        fig, axes = plt.subplots(nrows=1, ncols=1)
        axes.plot(np.real(self.FreqWave))
        axes.set_title('Real Freq Wave')
        plt.show()

    def plotMask(self):
        fig, axes = plt.subplots(nrows = 1, ncols = 1)
        axes.plot(self.mask)
        axes.set_title('Frequency Mask')
        plt.show()

    def plotFFT(self):
        fig, axes = plt.subplots(nrows = 1, ncols = 2)
        axes[0].plot(self.FFT_amp)
        axes[0].set_title('FFT Amp')
        axes[1].plot(self.FFT_phi)
        axes[1].set_title('FFT Phi')
        plt.tight_layout()
        plt.show()

    def extract_image(self):
        x, y = self.center
        y_box, x_box = self.boxradius
        self.image = self.original_image[(y - y_box):(y + y_box),(x - x_box):(x + x_box)]

    def disp_subI(self):
        plt.imshow(self.image, cmap='grey')
        plt.show()

    