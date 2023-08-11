import numpy as np
from motion_detection.IMotionDetector import IMotionDetector
import cv2

import warnings
warnings.filterwarnings("error")

MAX_FRAME_RESOLUTION_FLOW = 130_000

class FlowMotionDetector(IMotionDetector):
    def __init__(self, sample_frame: np.ndarray, *, 
                 grid_shape: tuple[int]=(7,7),
                 detection_threshold: float=0.1,
                 object_threshold: float=0.05
                 ):
        '''
        
        Parameters
        ----------
        grid_shape: tuple[int]
            step on x and y-axis when iterating through flow frame in search of figure
        
        '''
        self.new_size = None
        self.old_size = list(reversed(sample_frame.shape[:2]))

        H, W = sample_frame.shape[:2]

        if H * W > MAX_FRAME_RESOLUTION_FLOW:
            # h * w < MAX_FRAME_RESOLUTION_FLOW
            # H * W * a^2 < MAX_FRAME_RESOLUTION_FLOW
            # a^2 < MAX_FRAME_RESOLUTION_FLOW / H*W
            a = np.sqrt(MAX_FRAME_RESOLUTION_FLOW / (H * W))
            new_w, new_h = int(W * a), int(H * a)
            self.new_size = (new_w, new_h)
            sample_frame = cv2.resize(sample_frame, self.new_size)

        self.previous = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
        self.hsv = np.zeros_like(sample_frame)
        self.hsv[:, :, 1] = 255

        self.grid_shape = grid_shape
        assert 0 <= detection_threshold <= 1 and 0 <= object_threshold <= 1, "Thresholds must be in a range of [0; 1]"
        self.detection_threshold = detection_threshold
        self.object_threshold = object_threshold
    
    def detect_motion(self, frame: np.ndarray, return_processed_frame=False):
        if self.new_size is not None: 
            frame = cv2.resize(frame, self.new_size)
        
        calc = cv2.calcOpticalFlowFarneback(self.previous, frame, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        mag, ang = cv2.cartToPolar(calc[:, :, 0], calc[:, :, 1])
        self.hsv[:, :, 0] = ang * 180 / np.pi / 2
        self.hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

        self.previous = frame
        flow = cv2.bilateralFilter(flow, 20, 0.01, 1)
        # post-proceessing here (extract figures from flow)
        figures = self.__extract_figures(flow)

        if return_processed_frame:
            return figures, flow
        return figures
    
    def __extract_figures(self, flow: np.ndarray):
        H, W = flow.shape[:2]
        
        # perform grid checking looking for pixels that aren't close to estimated pixel 
        # (which could be calculated as mean of previous pixels or mean of some area around current pixel)
        step_x, step_y = self.grid_shape

        window_radius = 1

        difference_map = np.zeros(((H + step_y - 1) // step_y, (W + step_x - 1) // step_x))
        flow_median = np.median(cv2.cvtColor(flow, cv2.COLOR_BGR2GRAY))
        print(flow_median)        

        for y in range(window_radius * step_y, H-window_radius, step_y):
            for x in range(window_radius * step_x, W-window_radius, step_x):
                # calculate mean and compare to current pixel
                mean = flow[
                    y-window_radius*step_y : y+window_radius*step_y+1 : step_y, 
                    x-window_radius*step_x : x+window_radius*step_x+1 : step_x].mean(axis=(0,1))
                
                if np.array([.114, .587, .299]).dot(flow[y, x]) < flow_median:
                    flow[y, x] = mean
                    print(f'changed {y} {x}')

                difference_map[y // step_y, x // step_x] = self.__color_diff(mean, flow[y,x])
        
        difference_map /= 768 # normalization (max value of difference is ~ 767.83)
        self.detection_threshold = difference_map.max() * 2/6 + difference_map.min() * 4/6

        cv2.imshow('diff map', cv2.resize(difference_map, (difference_map.shape[1] * step_x, difference_map.shape[0] * step_y), interpolation=cv2.INTER_NEAREST))
        # now we need to detect contours in difference_map - white regions
        # firstly lets find some big values and then run bfs with smaller threshold from that pixel to expand contour
        rectangles = []
        for y in range(H // step_y):
            for x in range(W // step_x):
                # ignore black spots
                if difference_map[y, x] > self.detection_threshold:
                    rect = self.__inflateRectangle(difference_map, y, x)
                    rectangles.append(rect)
        if not rectangles:
            return np.array([], dtype='int32')
        rectangles = np.array(rectangles, dtype=float)

        rectangles[:, 0 :: 2] *= step_x # scale to flow size
        rectangles[:, 1 :: 2] *= step_y

        if self.new_size is not None:
            rectangles[:, 0 :: 2] *= self.old_size[0] / self.new_size[0] # scale to frame size
            rectangles[:, 1 :: 2] *= self.old_size[1] / self.new_size[1]

        return rectangles.astype('int32')
        
    def __color_diff(self, c1, c2):
        ''' colors in bgr '''
        c1 = c1.astype('int32')
        c2 = c2.astype('int32')
        rmean = ( c1[2] + c2[2] ) / 2

        b = c1[0] - c2[0]
        g = c1[1] - c2[1]
        r = c1[2] - c2[2]
        
        return np.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256))
    
    def __inflateRectangle(self, frame, start_y, start_x):
        '''Starting at (x,y), tries to find inner points of figure with BFS, and then fits it into rectangle'''
        rect = [start_x,start_y,start_x,start_y]

        visited = np.zeros_like(frame, dtype=bool)
        grid_height, grid_width = visited.shape

        queue = []

        visited[start_y, start_x] = True
        queue.append( (start_y, start_x) )

        steps = np.array(
            [[-1,  0], [1,  0], [0,  1], [0,  -1],
             [-1, -1], [1,  1], [1, -1], [-1,  1]]
        )

        in_bounds = lambda y, x: 0 <= y < grid_height and 0 <= x < grid_width
        self.object_threshold = frame.max() * 1/10 + frame.min() * 9/10
        while queue:
            y, x = queue.pop(0)
            in_figure = False

            # part of object
            if frame[y, x] > self.object_threshold:
                in_figure = True
                # update rect
                if x < rect[0]:
                    rect[0] = x
                if y < rect[1]:
                    rect[1] = y
                if x > rect[2]:
                    rect[2] = x
                if y > rect[3]:
                    rect[3] = y
            
            if not in_figure: 
                continue

            for step in steps:
                ny, nx = y+step[0], x+step[1]
                
                if in_bounds(ny, nx) and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append( (ny, nx) )

        return rect