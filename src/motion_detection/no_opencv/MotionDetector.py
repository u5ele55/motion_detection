from motion_detection.IMotionDetector import *
from collections import deque
from utilities.geometry import *

import cv2

class CustomMotionDetector(IMotionDetector):
    def __init__(self, *, 
                 capacity: int=10, 
                 min_height: int=12, 
                 min_width: int=12, 
                 avg_pooling_size: tuple[int]=(3, 3)):

        self.frame_pool = deque([])
        self.pool_sum = None

        self.step = (np.ceil(min_height / avg_pooling_size[0]), np.ceil(min_width / avg_pooling_size[1]))
        self.capacity = capacity

    def detect_motion(self, frame: np.ndarray, return_processed_frame: bool = False):
        if self.pool_sum is None:
            self.pool_sum = np.zeros(frame.shape, dtype='int32')

        if len(self.frame_pool) != self.capacity:
            self.frame_pool.append(frame.copy())
            self.pool_sum += frame
            if return_processed_frame:
                return np.array([]), np.abs(frame - self.pool_sum // len(self.frame_pool)).astype('uint8')
            return np.array([])
        
        processed = np.abs(frame - self.pool_sum // self.capacity)
        
        # Update mean
        self.pool_sum -= self.frame_pool.popleft()
        self.frame_pool.append(frame.copy())
        self.pool_sum += frame
        
        # look for white figures
        contours = self.__grid_contour_search(processed)

        return contours, cv2.threshold(processed.astype('uint8'), 3, 255, cv2.THRESH_BINARY)[1]
    
    def __grid_contour_search(self, frame):
        threshold = 30
        step_y = 10
        step_x = 10

        H,W = frame.shape[:2]

        contours = []

        for row in range(step_y, H, step_y):
            for col in range(step_x, W, step_x):
                if frame[row, col] >= threshold:
                    # ignore pixels inside already created rectangles
                    if self.__insideContour(contours, row, col):
                        continue

                    rect = self.__inflateRectangle(frame, row, col)
                    if (rect[2] - rect[0]) * (rect[3] - rect[1]) > 20:
                        contours.append((np.random.randint(0, 1000000), rect))
        
        return contours
    
    def __inflateRectangle(self, frame, start_y, start_x):
        '''Starting at (x,y), tries to find inner points of figure with BFS, and then fits it into rectangle'''
        rect = [start_x,start_y,start_x,start_y]
        step_y = 2
        step_x = 2
        threshold = 3

        H,W = frame.shape[:2]

        minkx, minky = -(start_x+step_x-1)//step_x, -(start_y+step_y-1)//step_y
        maxkx, maxky = (W-start_x-1)//step_x, (H-start_y-1)//step_y
        
        visited = np.zeros(
            (maxky - minky + 1, maxkx - minkx + 1), 
            dtype=bool)
        grid_height, grid_width = visited.shape
        queue = []

        visited[start_y // step_y, start_x // step_x] = True
        queue.append( (start_y // step_y, start_x // step_x) )

        steps = np.array(
            [[-1, 0], [1, 0], [0, 1], [0, -1]]
        )

        in_bounds = lambda y, x: 0 <= y < grid_height and 0 <= x < grid_width

        while queue:
            y, x = queue.pop(0)
            in_figure = False

            absolute_y, absolute_x = start_y % step_y + y * step_y, start_x % step_x + x * step_x

            if frame[absolute_y, absolute_x] > threshold:
                in_figure = True
                # update rect
                if absolute_x < rect[0]:
                    rect[0] = absolute_x
                if absolute_y < rect[1]:
                    rect[1] = absolute_y
                if absolute_x > rect[2]:
                    rect[2] = absolute_x
                if absolute_y > rect[3]:
                    rect[3] = absolute_y
            
            if not in_figure: 
                continue

            for step in steps:
                ny, nx = y+step[0], x+step[1]
                
                if in_bounds(ny, nx) and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append( (ny, nx) )

        return rect

    def __insideContour(self, contours, y, x):
        for _, rect in contours:
            if Geometry.inside((x,y), rect): 
                return True
        
        return False
