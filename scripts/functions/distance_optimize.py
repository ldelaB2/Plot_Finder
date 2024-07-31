import numpy as np
from shapely.geometry import Polygon, Point
from collections import deque
from functions.display import disp_spiral_path
from matplotlib import pyplot as plt


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
        if not test_points:
             return False
        
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

def find_valid_points(working_list, new_rect, radi, rect_dict):
        nbrs_to_check = list(set(rect_dict[new_rect].neighbors).intersection(set(working_list)))
        if not nbrs_to_check:
            valid = True
            rect_dict[new_rect].valid_points = rect_dict[new_rect].template_points

        else:
            intersection = None
            for nbr in nbrs_to_check:
                estimated_center = rect_dict[nbr].predict_neighbor_position(new_rect)
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

def back_track(working_list, rect_dict, stop_indx):
        # Stop if we reach the starting point
        if len(working_list) == stop_indx:
            return None
        
        last_rect = working_list[-1]
        move_flag = rect_dict[last_rect].move_center()

        if not move_flag:
            working_list.pop()
            return back_track(working_list, rect_dict, stop_indx)
        
        else:
            return working_list

def optimize_placement(path, radi, rect_dict, max_iter = 1000, existing_list = None):
        working_list = []

        if existing_list is not None:
            for rect in existing_list:
                working_list.append((rect.range, rect.row))

            zero_pad = [(0,0)] * len(working_list)
            path = zero_pad + path
        else:
            working_list.append(path[0])

        stop_indx = len(working_list)
        new_indx = len(working_list)
        cnt = 0
        success = True

        while new_indx < len(path):
            cnt += 1
            if cnt > max_iter:
                print("Max Iteration Reached")
                success = False
                break

            new_rect = path[new_indx]
            valid = find_valid_points(working_list, new_rect, radi, rect_dict)
            if valid:
                rect_dict[new_rect].find_center()
                working_list.append(new_rect)
                new_indx += 1

            else:
                working_list = back_track(working_list, rect_dict, stop_indx)
                if working_list is None:
                    print("No valid placement found")
                    success = False
                    break
                else:
                    new_indx = len(working_list)

        return success

def compute_score(rect_list):
        score = 0
        for rect in rect_list:
            if rect.score is not None:
                score += rect.score

        score = score / len(rect_list)
        return score

def bfs_distance(rect_list, param_dict, model, start_point = None):
    # Pull the param
    valid_radi = param_dict['valid_radi']
    neighbor_radi = param_dict['neighbor_radi']
    x_radi = param_dict['x_radi']
    y_radi = param_dict['y_radi']

    if start_point is None:
        start_point = compute_start_point(rect_list)
    
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

def compute_start_point(rect_list):
    ranges = np.unique([rect.range for rect in rect_list])
    rows = np.unique([rect.row for rect in rect_list])
    start_rng = np.median(ranges).astype(int)
    start_row = np.median(rows).astype(int)
    return (start_rng, start_row)
    
def bfs_add_rect(new_list, existing_list, model, param_dict):
    # Pull the params
    neighbor_radi = param_dict['neighbor_radi']
    valid_radi = param_dict['valid_radi']
    x_radi = param_dict['x_radi']
    y_radi = param_dict['y_radi']

    # Compute the template scores
    for rect in new_list:
        rect.compute_template_score(model, x_radi, y_radi)

    # Compute initial neighbors for path
    compute_neighbors(new_list, neighbor_radi)
    start_point = compute_start_point(new_list)
    graph = create_graph(new_list)

    # Compute the bfs path
    path = bfs(graph, start_point, length = len(new_list))

    # Recompute the neighbors
    compute_neighbors(new_list, neighbor_radi, existing_list)

    # Create the rect dictionary
    rect_dict = {(rect.range, rect.row): rect for rect in new_list}
    for rect in existing_list:
        rect_dict[(rect.range, rect.row)] = rect

    sucess_flag = False
    while not sucess_flag:
        # Optimize the placement
        sucess_flag = optimize_placement(path, valid_radi, rect_dict)
        if sucess_flag:
            print("Found Valid Placement")
            score = compute_score(new_list)
            print(f"Final Score {score}")
        else:
            print("Increasing Valid Radius")
            valid_radi += 1


    return new_list, score

def compute_neighbors(rect_list, neighbor_radi, existing_list = None):
    # Find min and max
    ranges = np.array([rect.range for rect in rect_list])
    rows = np.array([rect.row for rect in rect_list])

    # Add existing rectangles if needed
    if existing_list is not None:
        existing_ranges = np.array([rect.range for rect in existing_list])
        existing_rows = np.array([rect.row for rect in existing_list])
        ranges = np.concatenate((ranges, existing_ranges))
        rows = np.concatenate((rows, existing_rows))

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
        rect.update_neighbor_position()

    return

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

