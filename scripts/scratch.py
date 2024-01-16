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


# Find Contors
        contours, _ = cv.findContours(obj_filtered_wavepad, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        tmp = np.zeros_like(obj_filtered_wavepad)
        tmp = cv.drawContours(tmp, contours, -1, 1,10)


        # Breaking up the skeleton
        binary_matrices = list(itertools.product([0, 1], repeat=9))
        kernels = [truple for truple in binary_matrices if truple.count(1) > 3]

        print(f"Using {ncore} cores to prune range wavepad")
        with multiprocessing.Pool(processes=ncore) as pool:
            results = pool.map(
                break_up_skel, [(kernel, skel, [3, 3]) for kernel in kernels]
            )

        unique_points = [tuple(point) for sublist in results if sublist is not None for point in sublist]
        unique_points = np.array(list(set(unique_points)))
        broken_skel = skel.astype(np.uint8)
        broken_skel[unique_points[:, 0], unique_points[:, 1]] = 0

        # Filtering out objects under 300 pixels
        num_objects, broken_skel_labeled, img_stats, _ = cv.connectedComponentsWithStats(broken_skel)
        object_areas = img_stats[:, 4]
        area_thresh = 300
        obj_to_remove = np.where(object_areas < area_thresh)
        broken_skel_labeled[np.isin(broken_skel_labeled, obj_to_remove)] = 0
        broken_skel_labeled[broken_skel_labeled > 0] = 1


#Dialating to fill holes
        num_iterations = 20
        tmp = broken_skel_labeled.astype(np.uint8)
        kernel1 = np.ones((10,50))
        for e in range(num_iterations):
            tmp = cv.dilate(tmp, kernel1)

        tmp[:,:(min_index + 1)] = 0
        tmp[:, max_index:] = 0
        skel = skeletonize(tmp.astype(bool))








        search_area = (145, 600)
        rectangles = []
        flag1 = True
        flag2 = True
        cnt = -1
        while flag1:
            if flag2:
                top_left = cp[0,0:2]
                cp = cp[1:, :]

                points = cp[:,0:2]

                test, _ = eludian_distance(top_left, points, return_points=True)


                xmin, xmax = top_left[1] * .75, top_left[1] + search_area[0]
                ymin, ymax = top_left[0] * .75, top_left[0] + search_area[1]

                valid_points = cp[(cp[:,1] >= xmin) & (cp[:,1] <= xmax) & (cp[:,0] >= ymin) & (cp[:,0] <= ymax), 0:2]
                valid_points = np.vstack((valid_points, top_left))
                rectangles.append(valid_points)

                flag2 = False
                cnt += 1

                if valid_points.shape[0] < 4:
                    flag1 = False
            else:
                top_left = rectangles[cnt][0]
                indx = np.where((cp[:,0] == top_left[0]) & (cp[:,1] == top_left[1]))

                cp = np.delete(cp, indx, axis = 0)

                xmin, xmax = top_left[1] * .75, top_left[1] + search_area[0]
                ymin, ymax = top_left[0] * .75, top_left[0] + search_area[1]

                valid_points = cp[(cp[:, 1] >= xmin) & (cp[:, 1] <= xmax) & (cp[:, 0] >= ymin) & (cp[:, 0] <= ymax),
                               0:2]
                valid_points = np.vstack((valid_points, top_left))
                rectangles.append(valid_points)

                cnt += 1
                if valid_points.shape[0] < 4:
                    flag2 = True

        plt.scatter(cp[:,1],cp[:,0], color = 'red', marker='o')

        cp_y_sorted = cp[np.argsort(cp[:,0])]
        cp_x_sorted = cp[np.argsort(cp[:, 1])]

        def break_up_skel(args):
            kernel, skel, shape = args
            kernel = np.array(kernel).reshape(shape[0], shape[1]).astype(np.uint8)
            tmp = np.argwhere(cv.erode(skel.astype(np.uint8), kernel, iterations=1).astype(bool))
            if tmp.size > 0:
                tmp = tmp.tolist()
                return tmp
            else:
                return

    def centerline_from_contour(contours, poly_degree):
        centerlines = []
        for contour in contours:
            # Rescaling data
            x_mean = np.mean(contour[:, 0, 0])
            y_mean = np.mean(contour[:, 0, 1])
            x_std = np.std(contour[:, 0, 0])
            y_std = np.std(contour[:, 0, 1])
            x_scaled = (contour[:, 0, 0] - x_mean) / x_std
            y_scaled = (contour[:, 0, 1] - y_mean) / y_std

            # Fitting polynomial
            polyfit_coefficients = np.polyfit(x_scaled, y_scaled, poly_degree)

            # Find min and max Y values of the contour
            min_y = np.min(contour[:, 0, 1])
            max_y = np.max(contour[:, 0, 1])

            # Rescale and find X values the return to original scale
            y_vals = np.arange(min_y, max_y, 1)
            y_vals_scaled = (y_vals - y_mean) / y_std
            x_vals_scaled = np.polyval(polyfit_coefficients, y_vals_scaled)
            x_vals = (x_vals_scaled * x_std + x_mean).astype(int)

            # Generating the output
            center_line = list(zip(y_vals, x_vals))
            centerlines.append(center_line)

        return centerlines
    
    
       def compute_rectangles(self):
        cp = eludian_distance((0,0),self.cp, return_points = True)

        # Sorting the points into ranges
        sorted_points = []

        while True:
            start_point = cp[0,:]
            cp = cp[1:, :]

            temp_range_points = []
            temp_range_points.append(start_point)
            flag = [False]
            find_points(start_point, cp, temp_range_points, flag)
            temp_range_points = np.array(temp_range_points)
            sorted_points.append(temp_range_points)
            if flag[0]:
                break
            else:
                set1 = set(map(tuple, temp_range_points))
                set2 = set(map(tuple, cp))
                new_cp = set2.difference(set1)
                new_cp = np.array(list(new_cp))
                cp = eludian_distance((0, 0), new_cp, return_points=True)

        # Finding the rectangles
        for e in range(len(sorted_points) - 1):
            top_points = sorted_points[e]
            bottom_points = sorted_points[e + 1]

            if top_points.shape[0] != bottom_points.shape[0]:
                print("Warning Num points in top and bottom not equal")
            else:
                for k in range(top_points.shape[0] - 1):
                    top_left = top_points[k,:]
                    top_right = top_points[k + 1,:]
                    bottom_left = bottom_points[k,:]
                    bottom_right = bottom_points[k + 1,:]
                    points = [top_left, top_right, bottom_left, bottom_right]
                    self.four_2_five_rect(points)

    def four_2_five_rect(self, points):
        top_left, top_right, bottom_left, bottom_right = points
        w1 = np.linalg.norm(top_left - top_right)
        w2 = np.linalg.norm(bottom_left - bottom_right)
        h1 = np.linalg.norm(top_left - bottom_left)
        h2 = np.linalg.norm(top_right - bottom_right)
        width = ((w1 + w2)/4).astype(int)
        height = ((h1 + h2)/4).astype(int)
        center = np.mean((top_left,top_right,bottom_left,bottom_right), axis = 0).astype(int)
        rect = (center, width, height, 0)
        self.rect_list.append(rect)

    def disp_rectangles(self, img):
        plt.close('all')
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        for rect in self.rect_list:
            bottom_left = rect[0] - (rect[2], rect[1])
            bottom_left = bottom_left[::-1]
            rect_patch = patches.Rectangle(bottom_left, (rect[1] * 2), (rect[2] * 2), linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect_patch)
            
        ax.set_axis_off()
        return fig
     

     def find_points(start_point, points, point_list, flag):
    next_point = eludian_distance(start_point, points, True)[0, :]
    if (np.linalg.norm(next_point - start_point).astype(int)) > 150:
        return
    else:
        point_list.append(next_point)
        indx = np.where((points[:, 0] == next_point[0]) & (points[:, 1] == next_point[1]))
        points = np.delete(points, indx, axis=0)
        if points.shape[0] == 1:
            point_list.append(points[0,:])
            flag[0] = True
            return
        else:
            find_points(next_point, points, point_list, flag)


def eludian_distance(target, points, return_points=False):
    target = np.array(target)
    points = np.array(points)
    dist = np.sqrt(np.sum((target - points) ** 2, axis=1))
    if return_points:
        indx = np.argsort(dist)
        points = points[indx]
        return points
    else:
        return dist