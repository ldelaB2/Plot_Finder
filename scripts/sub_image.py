import numpy as np
import cv2 as cv
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
from functions import bindvec, findmaxpeak

class sub_image:
    def __init__(self, image, boxradius, center):
        """
        Initializes the sub_image object.

        Parameters:
            image (ndarray): The original image.
            boxradius (tuple): The radius of the box to extract from the original image.
            center (tuple): The center of the box.

        Attributes Created:
            self.original_image: The original image.
            self.boxradius: The radius of the box to be used in the scatter path.
            self.center: The center of the box.
            self.axis: The axis of the image. Initialized to 0.
                0 is used to compute over rows
                1 is used to compute over ranges

        It first stores the input parameters as attributes.
        Then, it calls the extract_image method to extract the sub-image from the original image.
        Finally, it sets the original_image attribute to None to free up memory.
        """
        self.original_image = image
        self.boxradius = boxradius
        self.center = center
        self.extract_image()
        self.original_image = None
        self.axis = 0

    def extract_image(self):
        """
        This method extracts a sub-image from the original image.

        Attributes Modified:
            self.image: The extracted sub-image.

        It first gets the center of the sub-image and the radius of the box to be extracted.
        Then, it uses these values to slice the original image and extract the sub-image.
        Finally, it stores the extracted sub-image in the image attribute.
        """
        x, y = self.center
        y_box, x_box = self.boxradius
        self.image = self.original_image[(y - y_box):(y + y_box),(x - x_box):(x + x_box)]

    def phase2(self,FreqFilterWidth, direction, mask, expand_radi):
        """
        This method performs the second phase of the image processing.

        Parameters:
            FreqFilterWidth (int): The width of the frequency filter to be used.
            direction (int): The direction of the processing. 0 for rows, 1 for ranges.
            mask (ndarray): The mask to be used in the frequency filtering.
            num_pixels (int): The number of pixels to be used in the calculation of the pixel value.

        Attributes Modified:
            self.axis: The direction of the processing. Set to the input direction.

        Returns:
            self.pixelval: The calculated pixel value.

        It first sets the direction of the processing.
        Then, it computes the FFT of the sub-image, filters the FFT using the mask, generates the frequency wave, converts the frequency wave to spatial domain, and calculates the pixel value.
        """
        self.axis = direction
        self.computeFFT()
        self.filterFFT(mask, FreqFilterWidth)
        self.generateWave()
        self.convertWave2Spacial()
        
        tindex = 1 - self.axis
        pixelval = self.SpacialWave[(self.boxradius[tindex] - expand_radi):(self.boxradius[tindex] + expand_radi + 1)]

        return pixelval

    def phase1(self, FreqFilterWidth):
        """
        This method performs the first phase of the image processing.

        Parameters:
            FreqFilterWidth (int): The width of the frequency filter to be used.
            DispOutput (bool, optional): If True, it will display the output graphs.

        Attributes Modified:
            self.axis: The direction of the processing. Set to 0 for rows and 1 for ranges.

        Returns:
            row_wave (ndarray): The frequency wave for the rows.
            range_wave (ndarray): The frequency wave for the columns.

        It first sets the direction of the processing to rows, computes the FFT of the sub-image, filters the FFT, generates the frequency wave, and stores the absolute value of the frequency wave.
        Then, it sets the direction of the processing to ranges and repeats the same steps.
        If DispOutput is True, it displays the sub-image, the FFT, the mask, and the frequency wave after each set of steps.
        """
         
        #Processing rows
        self.axis = 0
        self.computeFFT()
        self.filterFFT(None, FreqFilterWidth)
        self.generateWave()
        row_wave = abs(self.FreqWave)

        #Processing ranges
        self.axis = 1
        self.computeFFT()
        self.filterFFT(None, FreqFilterWidth)
        self.generateWave()
        range_wave = abs(self.FreqWave)

        return row_wave, range_wave

    def computeFFT(self):
        """
        This method computes the Fast Fourier Transform (FFT) of the sub-image.

        Attributes Modified:
            self.FFT_amp: The amplitude of the FFT.
            self.FFT_phi: The phase of the FFT.

        It first computes the mean of the sub-image along the specified axis and subtracts the overall mean to center the data around zero.
        Then, it computes the FFT of the centered data and shifts the zero-frequency component to the center of the spectrum.
        Finally, it calculates the amplitude and phase of the FFT and stores them in the FFT_amp and FFT_phi attributes, respectively.
        Bindvec is used to convert the amplitude to a vector with min 0 and max 1
        """
        # Dir(1) = rows    Dir(0) = columns
        # Computing fft
        sig = np.mean(self.image, axis = self.axis)
        fsig = fft(sig - np.mean(sig))
        fsig = fftshift(fsig)
        self.FFT_amp = bindvec(abs(fsig))
        self.FFT_phi = np.angle(fsig)

    def filterFFT(self, mask, FreqFilterWidth):
        """
        This method filters the Fast Fourier Transform (FFT) of the sub-image.

        Parameters:
            mask (ndarray): The mask to be used in the frequency filtering.
            FreqFilterWidth (int): The width of the frequency filter to be used.

        Attributes Modified:
            self.mask: The dilated mask used for filtering.

        It first finds the maximum peak of the FFT amplitude within the mask.
            if mask is None then it uses the entire FFT amplitude.
        Then, it creates a new mask with the same shape as the FFT amplitude and sets the value at the maximum peak to 1.
        It dilates the mask using a circular kernel with the specified frequency filter width.
        Finally, it stores the dilated mask in the mask attribute.
        """
        #Finding max peak
        max_peak = findmaxpeak(self.FFT_amp, mask)
        # Create Mask
        mask = np.zeros_like(self.FFT_amp)
        mask[max_peak] = 1
        # Dilate Mask
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (FreqFilterWidth,FreqFilterWidth))
        self.mask = cv.dilate(mask, kernel, iterations = 1).flatten()

    def generateWave(self):
        """
        This method generates the frequency wave from the filtered Fast Fourier Transform (FFT) of the sub-image.

        Attributes Modified:
            self.FreqWave: The generated frequency wave.

        It first multiplies the FFT amplitude and phase by the mask to filter out the frequencies outside the mask.
        Then, it combines the filtered amplitude and phase into a complex number to form the frequency wave.
        Finally, it stores the frequency wave in the FreqWave attribute.
        """
        #Computing Frequency Wave
        amp = self.FFT_amp * self.mask
        phi = self.FFT_phi * self.mask
        self.FreqWave = amp * np.exp(1j * phi)

    def convertWave2Spacial(self):
        """
        This method converts the frequency wave to the spatial domain.

        Attributes Modified:
            self.SpacialWave: The wave in the spatial domain.

        It first shifts the zero-frequency component of the frequency wave to the beginning of the spectrum.
        Then, it computes the inverse Fast Fourier Transform (IFFT) of the shifted frequency wave to convert it to the spatial domain.
        Finally, it takes the real part of the converted wave and stores it in the SpacialWave attribute.
        """
        # Compute Spacial Wave
        self.SpacialWave = bindvec(np.real(ifft(fftshift(self.FreqWave))))


    def plotSpacialWave(self):
        """
        This method plots the spatial wave.

        It creates a new figure and axes using matplotlib's subplots function.
        Then, it plots the spatial wave on the axes and sets the title of the axes to 'Spacial Wave'.
        Finally, it displays the figure using matplotlib's show function.
        """
        fig, axes = plt.subplots(nrows=1, ncols=1)
        axes.plot(self.SpacialWave)
        axes.set_title('Spacial Wave')
        plt.show()

    def plotFreqWave(self):
        """
        This method plots the real part of the frequency wave.

        It creates a new figure and axes using matplotlib's subplots function.
        Then, it plots the real part of the frequency wave on the axes and sets the title of the axes to 'Real Freq Wave'.
        Finally, it displays the figure using matplotlib's show function.
        """
        fig, axes = plt.subplots(nrows=1, ncols=1)
        axes.plot(np.real(self.FreqWave))
        axes.set_title('Real Freq Wave')
        plt.show()

    def plotMask(self):
        """
        This method plots the mask used in the frequency filtering.

        It creates a new figure and axes using matplotlib's subplots function.
        Then, it plots the mask on the axes and sets the title of the axes to 'Frequency Mask'.
        Finally, it displays the figure using matplotlib's show function.
        """
        fig, axes = plt.subplots(nrows = 1, ncols = 1)
        axes.plot(self.mask)
        axes.set_title('Frequency Mask')
        plt.show()

    def plotFFT(self):
        """
        This method plots the amplitude and phase of the Fast Fourier Transform (FFT).

        It creates a new figure and two axes using matplotlib's subplots function.
        Then, it plots the FFT amplitude on the first axes and sets the title of the axes to 'FFT Amp'.
        It plots the FFT phase on the second axes and sets the title of the axes to 'FFT Phi'.
        It adjusts the padding between and around the subplots using matplotlib's tight_layout function.
        Finally, it displays the figure using matplotlib's show function.
        """
        fig, axes = plt.subplots(nrows = 1, ncols = 2)
        axes[0].plot(self.FFT_amp)
        axes[0].set_title('FFT Amp')
        axes[1].plot(self.FFT_phi)
        axes[1].set_title('FFT Phi')
        plt.tight_layout()
        plt.show()

    

    def disp_subI(self):
        """
        This method displays the extracted sub-image.

        It uses matplotlib's imshow function to display the sub-image in grayscale.
        Then, it uses matplotlib's show function to display the figure.
        """
        plt.imshow(self.image, cmap='grey')
        plt.show()

    