import numpy as np
from matplotlib import pyplot as plt

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

