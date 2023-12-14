def process_phase2_1core(self, FreqFilterWidth, row_mask, range_mask, num_pixles):
    raw_wavepad = np.zeros((self.num_points, 2))
    for e in range(self.num_points):
        center = self.path[e]
        subI = sub_image(self.image, self.boxradius, center)
        raw_wavepad[e, 0] = subI.phase2(FreqFilterWidth, 0, row_mask, num_pixles)
        raw_wavepad[e, 1] = subI.phase2(FreqFilterWidth, 1, range_mask, num_pixles)
        # Counter
        if e % 100 == 0:
            print(f"{((e / self.num_points) * 100):.1f} % Done")

    return (raw_wavepad)


 # Testing

    test = sparse_grid.path[100]
    test_sub_I = sub_image(g, boxradius, test)
    test_sub_I.disp_subI()
    test_sub_I.axis = 0
    test_sub_I.computeFFT()
    test_sub_I.plotFFT()
    test_sub_I.filterFFT(None, FreqFilterWidth_SparseGrid)
    test_sub_I.plotMask()
    test_sub_I.generateWave()
    test_sub_I.plotFreqWave()
    test_sub_I.convertWave2Spacial()
    test_sub_I.plotSpacialWave()
    test_sub_I.calcPixelValue(1, False)
    print(test_sub_I.pixelval)
    test_sub_I.calcPixelValue(0,False)
    print(test_sub_I.pixelval)


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




