from motion_detection.IMotionDetector import *
from collections import deque
from utilities.geometry import *

import cv2

class CustomMotionDetector(IMotionDetector):
    def __init__(self, *, 
                 capacity: int=10, 
                 
                 max_deviation: int = 0, 
                 patience: int = 2,
                 max_elapsed_time: int = 5,

                 move_threshold: int = 30,
                 object_threshold: int = 5,

                 detection_step: tuple[int]=(10,10),
                 object_selection_step: tuple[int]=(3,3)
                 ):
        '''
        
        Parameters
        ----------
        capacity : int
            quantity of previous frames which are used to detect motion on a new frame
        max_deviation : int
            maximal deviation of the center of an object relatively to its rectangle borders (in L1 norm) compared to previous frame.
        patience : int
            quantity of frames to wait to admit figure presence on frame
        max_elapsed_time : int
            quantity of frames to wait for previously moving object to admit its stopped
        
        move_threshold : int
            minimal brightness of difference of object and previous frames to consider
        object_threshold : int
            minimal value of the same difference when discovering object volume
        
        detection_step : tuple[int]
            step on x and y axis to perform when looking for objects
        object_selection_step : tuple[int]
            step on x and y axis to perform when discovering object volume
        '''
        self.capacity = capacity
        self.frame_pool = deque([])
        self.pool_sum = None

        self.figures = {}    # {rect_id: [vertices]}
        self.figure_center_streak = {} # {rect_id: [center_streak, elapsed_time, found_on_frame]}
        self.figure_cnt = 0

        self.max_deviation = max_deviation
        self.patience = patience
        self.max_elapsed_time = max_elapsed_time

        self.move_threshold = move_threshold
        self.object_threshold = object_threshold
        self.detection_step = detection_step
        self.object_selection_step = object_selection_step

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

        # attach ids to contours
        used_rectangles = [False] * len(contours)

        if not self.figures:
            for rect in contours:
                self.figures[self.figure_cnt] = rect.copy()
                self.figure_center_streak[self.figure_cnt] = [0, 0, True]
                self.figure_cnt += 1
        else:
            for i in range(len(contours)):
                inflated_rectangle = [
                    contours[i][0]-self.max_deviation, contours[i][1]-self.max_deviation,
                    contours[i][2]+self.max_deviation, contours[i][3]+self.max_deviation,
                    ]
                # find id s.t. self.figures[id] has center in contours[i] borders (consider deviation) 
                for id in self.figures:
                    self.figure_center_streak[id] = self.figure_center_streak.get(id, [0, 0, False])
                    self.figure_center_streak[id][2] = False
                    if Geometry.inside(Geometry.get_center(self.figures[id]), inflated_rectangle):
                        self.figures[id] = contours[i].copy()
                        used_rectangles[i] = True
                        self.figure_center_streak[id][0] += 1
                        self.figure_center_streak[id][1] = 0
                        self.figure_center_streak[id][2] = True
                        break
            # delete figures that wasn't found on frame
            to_del = []
            for id in self.figure_center_streak:
                if not self.figure_center_streak[id][2]:
                    # figure wasn't found
                    self.figure_center_streak[id][1] += 1
                    if self.figure_center_streak[id][1] >= self.max_elapsed_time:
                        to_del.append(id)
            for id in to_del:
                del self.figures[id]
                del self.figure_center_streak[id]
        
        # iterate through non-used rectangles and add them to self.figures
        for i in range(len(contours)):
            if not used_rectangles[i]:
                self.figures[self.figure_cnt] = contours[i].copy()
                self.figure_center_streak[self.figure_cnt] = [0, 0, True]
                self.figure_cnt += 1

        result = \
            [[i, self.figures[i]] for i in self.figures if self.figure_center_streak[i][0] >= self.patience]
        if return_processed_frame:
            return result, processed.astype('uint8')
        return result
    
    def __grid_contour_search(self, frame):
        step_y = self.detection_step[1]
        step_x = self.detection_step[0]

        H,W = frame.shape[:2]

        contours = []

        for row in range(step_y, H, step_y):
            for col in range(step_x, W, step_x):
                if frame[row, col] >= self.move_threshold:
                    # ignore pixels inside of already created rectangles
                    if self.__insideContour(contours, row, col):
                        continue

                    rect = self.__inflateRectangle(frame, row, col)
                    # TODO: change it to something more reasonable
                    if (rect[2] - rect[0]) * (rect[3] - rect[1]) > 20:
                        contours.append(rect)

        return contours
    
    def __inflateRectangle(self, frame, start_y, start_x):
        '''Starting at (x,y), tries to find inner points of figure with BFS, and then fits it into rectangle'''
        rect = [start_x,start_y,start_x,start_y]
        step_y = self.object_selection_step[1]
        step_x = self.object_selection_step[0]

        H,W = frame.shape[:2]

        # solution to 0 <= start_x + kx*step_x <= W-1
        minkx, minky = (-start_x+step_x-1)//step_x, (-start_y+step_y-1)//step_y
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

        offset_y, offset_x = start_y % step_y, start_x % step_x

        while queue:
            y, x = queue.pop(0)
            in_figure = False

            absolute_y, absolute_x = offset_y + y * step_y, offset_x + x * step_x

            if frame[absolute_y, absolute_x] > self.object_threshold:
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
        for rect in contours:
            if Geometry.inside((x,y), rect): 
                return True
        
        return False
