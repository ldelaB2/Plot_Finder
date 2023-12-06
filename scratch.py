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


