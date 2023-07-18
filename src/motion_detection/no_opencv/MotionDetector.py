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
        # __grid_contour_search
        
        return [], processed.astype('uint8')
    
    def __grid_contour_search(self, frame):
        threshold = 30
        step_y = 2
        step_x = 2
        for row in frame[::step_y]:
            for px in row[::step_x]:
                if px >= threshold:
                    pass
        pass
