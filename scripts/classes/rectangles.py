import numpy as np
from PIL import Image

from functions.image_processing import extract_rectangle, five_2_four_rect
from functions.image_processing import create_unit_square
from functions.optimization import compute_score


class rectangle:
    def __init__(self, rect):
        self.center_y = rect[0].astype(int)
        self.center_x = rect[1].astype(int)
        self.width = rect[2].astype(int)
        self.height = rect[3].astype(int)
        self.theta = rect[4]

        if len(rect) == 7:
            self.range = rect[5]
            self.row = rect[6]
        else:
            self.range = None
            self.row = None

        self.img = None
        self.ID = None
        self.flagged = False
        self.added = False
        self.unit_sqr = None
        self.neighbors = None
        self.recompute_unit_sqr = True
        self.x_radi = None
        self.y_radi = None
        self.t_radi = None
        self.h_radi = None
        self.w_radi = None

    def clear(self):
        self.added = False
        self.unit_sqr = None
        self.neighbors = None
        self.recompute_unit_sqr = True

    def create_sub_image(self):
        if self.recompute_unit_sqr:
            self.unit_sqr = create_unit_square(self.width, self.height)
            self.recompute_unit_sqr = False

        sub_image = extract_rectangle(self.center_x, self.center_y, self.theta, self.width, self.height, self.unit_sqr, self.img)
        return sub_image
    
    def compute_corner_points(self):
        points = (self.center_x, self.center_y, self.width, self.height, self.theta)
        corner_points = five_2_four_rect(points)
        return corner_points
  
    def move_rectangle(self, dX, dY, dT):
        center_x = self.center_x + dX
        center_y = self.center_y + dY
        theta = self.theta + dT
        
        new_img = extract_rectangle(center_x, center_y, theta, self.width, self.height, self.unit_sqr, self.img)
        
        return new_img
    
    def shrink_rectangle(self, dwidth, dheight):
        new_width = self.width + dwidth
        new_height = self.height + dheight
        new_unit_sqr = create_unit_square(new_width, new_height)
        new_img = extract_rectangle(self.center_x, self.center_y, self.theta, new_width, new_height, new_unit_sqr, self.img)

        return new_img
    
    def optimize_xy(self, template_img):
        # Compute the top left corner
        corner_points = self.compute_corner_points()
        top_left = corner_points[-1]
        negative_dx = self.min_x - top_left[0]
        positive_dx = self.max_x - top_left[0]
        min_x = top_left[0] + negative_dx
        max_x = top_left[0] + positive_dx

        negative_dy = self.min_y - top_left[1]
        positive_dy = self.max_y - top_left[1]

        min_y = top_left[1] + negative_dy
        max_y = top_left[1] + positive_dy
       
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(template_img.shape[1], max_x)
        max_y = min(template_img.shape[0], max_y)

        final_negative_x = top_left[0] - min_x
        final_negative_y = top_left[1] - min_y

        sub_template_img = template_img[min_y:max_y, min_x:max_x]
        
        if sub_template_img.size == 0:
            score = np.inf
            self.flagged = True
            return
        
        score = np.min(sub_template_img)

        best_position = np.argwhere(sub_template_img == score)[0]
        original_center = np.array([final_negative_y, final_negative_x])

        delta = best_position - original_center

        self.center_x += delta[1]
        self.center_y += delta[0]
        self.score = score

        return

    def compute_template_position(self):
        if not self.flagged:  
            best_point = np.argwhere(self.template_score == self.score)[0]
            delta_x = best_point[1] - self.x_radi
            delta_y = best_point[0] - self.y_radi

            self.center_x += delta_x
            self.center_y += delta_y
        else:
            return

        return
        
    def compute_neighbor_position(self, neighbor_distance, neighbor):
        rng_away = abs(self.range - neighbor[0])
        row_away = abs(self.row - neighbor[1])

        if neighbor[0] < self.range:
            rng_away *= -1
        if neighbor[1] < self.row:
            row_away *= -1

        dx = neighbor_distance[1] * row_away
        dy = neighbor_distance[0] * rng_away
        
        theta = np.radians(self.theta)

        dx_rot = dx * np.cos(theta) - dy * np.sin(theta)
        dy_rot = dx * np.sin(theta) + dy * np.cos(theta)

        dx_rot = int(np.round(dx_rot))
        dy_rot = int(np.round(dy_rot))

        estimated_x = self.center_x + dx_rot
        estimated_y = self.center_y + dy_rot

        return [estimated_x, estimated_y], [self.theta, self.width, self.height]
    
    def compute_radi(self, x_radi = None, y_radi = None, t_radi = None, h_radi = None, w_radi = None):
        if x_radi is not None:
            x_radi = np.round(self.width * x_radi).astype(int)
            corner_points = self.compute_corner_points()
            top_left = corner_points[-1]
            self.min_x = top_left[0] - x_radi
            self.max_x = top_left[0] + x_radi

        if y_radi is not None:
            y_radi = np.round(self.height * y_radi).astype(int)
            corner_points = self.compute_corner_points()
            top_left = corner_points[-1]
            self.min_y = top_left[1] - y_radi
            self.max_y = top_left[1] + y_radi

        if t_radi is not None:
            self.min_t = self.theta - t_radi
            self.max_t = self.theta + t_radi

        if h_radi is not None:
            h_radi = np.round(self.height * h_radi).astype(int)
            self.min_height = self.height - h_radi
            self.max_height = self.height + h_radi
        
        if w_radi is not None:
            w_radi = np.round(self.width * w_radi).astype(int)
            self.min_width = self.width - w_radi
            self.max_width = self.width + w_radi

        return
    
    def optimize_t(self, model, method = "L2"):
        current_score = compute_score(model, self.create_sub_image(), method)

        # First pass
        thetas = np.arange(self.min_t - self.theta, self.max_t - self.theta + 1, .1)
        scores = []
        for theta in thetas:
            new_img = self.move_rectangle(0, 0, theta)
            score = compute_score(model, new_img, method)
            scores.append(score)

        best_score = np.min(scores)

        if best_score < current_score:
            best_theta = thetas[np.argmin(scores)]
            self.theta += best_theta
            self.recompute_unit_sqr = True
            output = 1
        else:
            output = 0

        return output
    
    def optimize_height(self, model, method = "L2"):
        current_score = compute_score(model, self.create_sub_image(), method)

        # First pass
        heights = np.arange(self.min_height - self.height, self.max_height - self.height + 1, 20)
        scores = []

        for height in heights:
            new_img = self.shrink_rectangle(0, height)
            score = compute_score(model, new_img, method)
            scores.append(score)

        # Second pass
        rough_best_height = heights[np.argmin(scores)]
        heights = np.arange(rough_best_height - 10, rough_best_height + 10, 1)
        scores = []

        for height in heights:
            new_img = self.shrink_rectangle(0, height)
            score = compute_score(model, new_img, method)
            scores.append(score)

        best_score = np.min(scores)

        if best_score < current_score:
            best_height = heights[np.argmin(scores)]
            self.height += best_height
            self.recompute_unit_sqr = True
            output = 1
        else:
            output = 0
        
        return output

    def optimize_width(self, model,  method = "L2"):
        current_score = compute_score(model, self.create_sub_image(), method)

        # First pass
        widths = np.arange(self.min_width - self.width, self.max_width - self.width + 1, 10)
        scores = []

        for width in widths:
            new_img = self.shrink_rectangle(width, 0)
            score = compute_score(model, new_img, method)
            scores.append(score)

        # Second pass
        rough_best_width = widths[np.argmin(scores)]
        widths = np.arange(rough_best_width - 5, rough_best_width + 5, 1)
        scores = []

        for width in widths:
            new_img = self.shrink_rectangle(width, 0)
            score = compute_score(model, new_img, method)
            scores.append(score)

        best_score = np.min(scores)
        if best_score < current_score:
            best_width = widths[np.argmin(scores)]
            self.width += best_width
            self.recompute_unit_sqr = True
            output = 1
        else:
            output = 0
        
        return output
    
    def save_rect(self, img, path):
        unit_sqr = create_unit_square(self.width, self.height)
        sub_image = extract_rectangle(self.center_x, self.center_y, self.theta, self.width, self.height, unit_sqr, img)
        sub_image = Image.fromarray(sub_image)
        sub_image.save(path)

        return




        


       
    
 
 
    