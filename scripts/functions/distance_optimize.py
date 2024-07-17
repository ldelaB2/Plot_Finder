import numpy as np
from shapely.geometry import Polygon, Point
from collections import deque
from functions.display import disp_spiral_path
from matplotlib import pyplot as plt

def distance_optimize(rect_list, kernel, existing_list = None, weight = .5, update = False):
    # Compute the spiral path
    spiral_path = compute_spiral_path(rect_list)

    #Compute the neighbors
    compute_neighbors(rect_list, existing_list, kernel)

    # Find the range and row values
    ranges = np.array([rect.range for rect in rect_list])
    rows = np.array([rect.row for rect in rect_list])
    output = []
    flagged_output = []

    for point in spiral_path:
        path_rng = point[0]
        path_row = point[1]
        indx = np.where((ranges == path_rng) & (rows == path_row))[0][0]
        current_rect = rect_list[indx]
        neighbors = current_rect.neighbors
        expected_centers = []
        for neighbor in neighbors:
            # Find the neighbor
            nbr_rng = neighbor[0]
            nbr_row = neighbor[1]
            neighbor_indx = np.where((ranges == nbr_rng) & (rows == nbr_row))[0][0]
            neighbor_rect = rect_list[neighbor_indx]

            # Find the expected center
            expected_center = find_expected_center(current_rect, neighbor_rect)
            expected_centers.append(expected_center)
        
        # Compute the geometric median
        new_center = geometric_median(expected_centers)
        new_center = np.round(new_center).astype(int)

        dx = new_center[0] - current_rect.center_x
        dy = new_center[1] - current_rect.center_y

        tmp_output = [current_rect.range, current_rect.row, dx, dy]

        if current_rect.flagged:
            flagged_output.append(tmp_output)
        else:
            output.append(tmp_output)

    if update:
        # Compute the estimated distance of unflagged rectangles
        estimated_distances = np.array([np.sqrt(dx**2 + dy**2) for _, _, dx, dy in output])

        mean_distance = np.mean(estimated_distances)
        std_distance = np.std(estimated_distances)
        threshold = mean_distance + 2 * std_distance
        flag_indx = np.argwhere(estimated_distances > threshold)

        # Update the rectangles
        if flag_indx is not None:
            print(f"Flagging {len(flag_indx)} rectangles")
            for indx in flag_indx:
                indx = indx[0]
                rng, row, dx, dy = output[indx]
                rect_indx = np.where((ranges == rng) & (rows == row))[0][0]
                rect_list[rect_indx].flagged = True
                rect_list[rect_indx].center_x += dx
                rect_list[rect_indx].center_y += dy
            
            output = np.delete(output, flag_indx, axis = 0)

    # Move flagged rectanlges to the expected center
    if flagged_output is not None:
        for rng, row, dx, dy in flagged_output:
            rect_indx = np.where((ranges == rng) & (rows == row))[0][0]
            rect_list[rect_indx].center_x += dx
            rect_list[rect_indx].center_y += dy

    # Move unflagged rectangles to the weighted center
    if output is not None:
        for rng, row, dx, dy in output:
            rect_indx = np.where((ranges == rng) & (rows == row))[0][0]
            rect_list[rect_indx].center_x += np.round(weight * dx).astype(int)
            rect_list[rect_indx].center_y += np.round(weight * dy).astype(int)
            
    print("Finished optimizing distance")
    return 
    

def bfs_distance(args):
    rect_list, param_dict, start_point = args

    def create_circle_polygon(center, radius, resolution = 100):
        theta = np.linspace(0, 2*np.pi, resolution)
        theta[-1] = 0
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)

        return Polygon(np.column_stack((x, y)))

    def create_graph(rect_list):
        graph = {}
        for rect in rect_list:
            key = (rect.range, rect.row)
            graph[key] = rect.neighbors

        return graph

    def bfs(graph, start, length):
        visited = set()
        queue = deque([start])
        path = []

        while queue and len(path) < length:
            node = queue.popleft()

            if node not in visited:
                path.append(node)
                visited.add(node) # Mark as visited
                
                # Enqueue the neighbors
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        return path

    def compute_valid_points(rect, polygon):
        bounds = np.round(polygon.bounds).astype(int)
        test_points = rect.template_points
        test_points = np.array(list(test_points))
        x_bool = (test_points[:,0] >= bounds[0]) & (test_points[:,0] <= bounds[2])
        y_bool = (test_points[:,1] >= bounds[1]) & (test_points[:,1] <= bounds[3])
        total_bool = x_bool & y_bool
        test_points = test_points[total_bool]

        valid_points = set()
        for point in list(test_points):
            if polygon.contains(Point(point)):
                valid_points.add(tuple(point))

        if not valid_points:
            return False
        else:
            rect.valid_points = valid_points
            return True

    def find_valid_points(working_list, new_rect, rect_dict, radi):
        nbrs_to_check = list(set(rect_dict[new_rect].neighbors).intersection(set(working_list)))
        if not nbrs_to_check:
            valid = True
            rect_dict[new_rect].valid_points = rect_dict[new_rect].template_points

        else:
            intersection = None
            for nbr in nbrs_to_check:
                estimated_center = rect_dict[nbr].nbr_position[new_rect]
                tmp_circle = create_circle_polygon(estimated_center, radi)
                if intersection is None:
                    intersection = tmp_circle
                else:
                    intersection = intersection.intersection(tmp_circle)

            if intersection.is_empty:
                valid = False
            else:
                valid_points = compute_valid_points(rect_dict[new_rect], intersection)
                if valid_points:
                    valid = True
                else:
                    valid = False
            
        return valid
    
    def back_track(working_list, rect_dict):
        # Stop if we reach the starting point
        if len(working_list) == 1:
            return None
        
        last_rect = working_list[-1]
        move_flag = rect_dict[last_rect].move_center()

        if not move_flag:
            working_list.pop()
            return back_track(working_list, rect_dict)
        
        else:
            return working_list
    
    def optimize_placement(path, radi, rect_dict, max_iter = 1000):
        working_list = []
        working_list.append(path[0])
        new_indx = 1
        cnt = 0
        success = True

        while new_indx < len(path):
            cnt += 1
            if cnt > max_iter:
                print("Max Iteration Reached")
                success = False
                break

            new_rect = path[new_indx]
            valid = find_valid_points(working_list, new_rect, rect_dict, radi)
            if valid:
                rect_dict[new_rect].find_center()
                working_list.append(new_rect)
                new_indx += 1

            else:
                working_list = back_track(working_list, rect_dict)
                if working_list is None:
                    print("No valid placement found")
                    success = False
                    break
                else:
                    new_indx = len(working_list)

        print("Valid Placement Found")
        return success

    def compute_score(rect_list):
        score = 0
        for rect in rect_list:
            if rect.score is not None:
                score += rect.score

        score = score / len(rect_list)
        return score
    
    # Pull the param
    valid_radi = param_dict['valid_radi']
    neighbor_radi = param_dict['neighbor_radi']
    x_radi = param_dict['x_radi']
    y_radi = param_dict['y_radi']
    model = param_dict['model']

    # Compute the neighbors
    compute_neighbors(rect_list, neighbor_radi)

    # Compute template scores
    for rect in rect_list:
        rect.compute_template_score(model, x_radi, y_radi)

    # Create the graph
    graph = create_graph(rect_list)

    # Compute the bfs path
    path = bfs(graph, start_point, length = len(rect_list))

    # Create the rect dictionary
    rect_dict = {(rect.range, rect.row): rect for rect in rect_list}

    # Optimize the placement
    sucess_flag = optimize_placement(path, valid_radi, rect_dict)

    # Compute the score
    score = compute_score(rect_list)
    
    return [sucess_flag, score, rect_list]

    


    


    



def compute_neighbors(rect_list, neighbor_radi):
    # Find min and max
    ranges = np.array([rect.range for rect in rect_list])
    rows = np.array([rect.row for rect in rect_list])
    max_range = np.max(ranges)
    min_range = np.min(ranges)
    max_row = np.max(rows)
    min_row = np.min(rows)
    for rect in rect_list:
        # Find self range and row
        rng = rect.range
        row = rect.row

        # Find the neighbors
        rng_neighbors = np.arange(rng - neighbor_radi, rng + neighbor_radi + 1)
        row_neighbors = np.arange(row - neighbor_radi, row + neighbor_radi + 1)

        # Clip the neighbors to valid values
        rng_neighbors = np.unique(np.clip(rng_neighbors, min_range, max_range))
        row_neighbors = np.unique(np.clip(row_neighbors, min_row, max_row))

        # Create the neighbor list
        x, y = np.meshgrid(rng_neighbors, row_neighbors)
        tmp_neighbors = np.column_stack((x.ravel(), y.ravel()))
        
        # Remove self from neighbors
        self_indx = np.where((tmp_neighbors[:,0] == rng) & (tmp_neighbors[:,1] == row))[0]
        tmp_neighbors = np.delete(tmp_neighbors, self_indx, axis = 0)
        tmp_neighbors = [tuple(nbr) for nbr in tmp_neighbors]
        rect.neighbors = tmp_neighbors
        rect.predict_neighbor_position()

    return

def compute_spiral_path(rect_list, center_point = None):
    def find_center(rect_list):
        # Get the range and row values
        ranges = np.array([rect.range for rect in rect_list])
        rows = np.array([rect.row for rect in rect_list])

        # Get the unique range and row values
        unique_ranges = np.unique(ranges)
        unique_rows = np.unique(rows)

        center_range = np.round(np.median(unique_ranges)).astype(int)
        center_row = np.round(np.median(unique_rows)).astype(int)

        indx = np.where((ranges == center_range) & (rows == center_row))[0][0]
        center_x = rect_list[indx].center_x
        center_y = rect_list[indx].center_y

        return [center_x, center_y]

    def compute_polar_coordinates(rect_list, center):
        # Compute the polar coordinates
        polar_coords = []
        for rect in rect_list:
            dx = rect.center_x - center[0]
            dy = rect.center_y - center[1]
            rng = rect.range
            row = rect.row
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            polar_coords.append([r, theta, rng, row])

        return polar_coords
    
    def sort_polar_coordinates(polar_coords):
        polar_coords = np.array(polar_coords)
        distance = polar_coords[:,0]
        angle = polar_coords[:,1]
        # Sort in descending order
        sorted_indx = np.lexsort((angle, distance))
        sorted_points = polar_coords[sorted_indx]
        rng_row = sorted_points[:,2:]
        rng_row = rng_row.astype(int)
        return rng_row

    if center_point is None:
        center_point = find_center(rect_list)
    
    polar_coords = compute_polar_coordinates(rect_list, center_point)
    path = sort_polar_coordinates(polar_coords)

    return path

def find_expected_center(current_rect, neighbor_rect):
    # pull the values
    cnt_rng = current_rect.range
    cnt_row = current_rect.row
    nbr_rng = neighbor_rect.range
    nbr_row = neighbor_rect.row
    nbr_center = np.array([neighbor_rect.center_x, neighbor_rect.center_y])

    # Find the distance between the rectangles
    rng_away = abs(cnt_rng - nbr_rng)
    row_away = abs(cnt_row - nbr_row)

    if nbr_row > cnt_row:
        row_away = -row_away
    if nbr_rng > cnt_rng:
        rng_away = -rng_away


    # Find the expected center
    dx = row_away * neighbor_rect.width
    dy = rng_away * neighbor_rect.height

    # Rotate the vector
    theta = np.radians(neighbor_rect.theta)
    dx_rot = dx * np.cos(theta) - dy * np.sin(theta)
    dy_rot = dx * np.sin(theta) + dy * np.cos(theta)

    # Round the values
    dx_rot = np.round(dx_rot).astype(int)
    dy_rot = np.round(dy_rot).astype(int)

    # Find the expected center
    expected_center = nbr_center + np.array([dx_rot, dy_rot])

    return expected_center

def geometric_median(points, weights=None, tol = 1e-2):
    points = np.asarray(points)
    if weights is None:
        weights = np.ones(len(points))
    else:
        weights = np.asarray(weights)
    
    guess = np.mean(points, axis = 0)

    while True:
        distances = np.linalg.norm(points - guess, axis=1)
        nonzero = (distances != 0)
        
        if not np.any(nonzero):
            return guess
        
        w = weights[nonzero] / distances[nonzero]
        new_guess = np.sum(points[nonzero] * w[:, None], axis=0) / np.sum(w)
        
        if np.linalg.norm(new_guess - guess) < tol:
            return new_guess
        
        guess = new_guess