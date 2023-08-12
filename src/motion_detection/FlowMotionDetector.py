import numpy as np
from motion_detection.IMotionDetector import IMotionDetector
from utilities.geometry import Geometry
import cv2
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

from scipy.signal import medfilt2d

import warnings
warnings.filterwarnings("error")

MAX_FRAME_RESOLUTION_FLOW = 130_000

class SearchState:
    none = 0
    increased = 1
    decreased = 2
    increased_second = 3

class FlowMotionDetector(IMotionDetector):
    def __init__(self, sample_frame: np.ndarray, *, 
                 grid_shape: tuple[int]=(7,7),
                 detection_threshold: float=0.3,
                 selection_threshold: float=2/10
                 ):
        '''
        
        Parameters
        ----------
        grid_shape: tuple[int]
            step on y and x-axis when iterating through flow frame in search of figure
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
        assert 0 <= detection_threshold <= 1 and 0 <= selection_threshold <= 1 
        self.detection_threshold = detection_threshold
        self.selection_threshold = selection_threshold
        
        self.frames_processed = 0
    
    def detect_motion(self, frame: np.ndarray, return_processed_frame=False):
        if self.new_size is not None: 
            frame = cv2.resize(frame, self.new_size)
        
        calc = cv2.calcOpticalFlowFarneback(self.previous, frame, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        mag, ang = cv2.cartToPolar(calc[:, :, 0], calc[:, :, 1])

        mag = medfilt2d(mag[::self.grid_shape[0], ::self.grid_shape[1]])
        ang = medfilt2d(ang[::self.grid_shape[0], ::self.grid_shape[1]])

        self.__show_scaled('magnitude filtered', mag / mag.max(), self.grid_shape[0])
        self.__show_scaled('angles filtered', ang / (2*np.pi), self.grid_shape[0])

        # self.hsv[:, :, 0] = ang * 180 / np.pi / 2
        # self.hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # flow = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

        self.previous = frame

        self.__find_by_magnitude(mag)
        self.__find_by_angle(ang, mag)

        self.frames_processed += 1
        
        if return_processed_frame:
            return [], ang
        return []
    
    def __find_by_magnitude(self, magn: np.ndarray):
        radius = 1
        changes = np.zeros_like(magn)
        for y in range(radius, magn.shape[0] - radius):
            for x in range(radius, magn.shape[1] - radius):
                mean = magn[y-radius:y+radius+1,x-radius:x+radius+1].mean() 
                changes[y,x] = abs(magn[y,x] - mean)  # maybe division? 
        
        self.__show_scaled('changes magn', changes, 5)
        vars = np.zeros_like(changes)
        eps = 0.2
        
        # find figures by "increasing, decresing, increasing" factor
        for y in range(radius, magn.shape[0] - radius):
            state = SearchState.increased
            last_mean = 5
            for x in range(magn.shape[1]):
                mean = magn[y-radius:y+radius+1, x].mean()
                if mean < last_mean - eps:
                    if state == SearchState.increased:
                        state = SearchState.decreased
                    elif state == SearchState.increased_second:
                        state = SearchState.none
                if mean > last_mean + eps:
                    if state == SearchState.none:
                        state = SearchState.increased
                    elif state == SearchState.decreased:
                        state = SearchState.increased_second
                if x == magn.shape[1] - 1 and state == SearchState.decreased:
                    state = SearchState.increased_second

                if state == SearchState.increased_second:
                    # run figure extraction
                    vars[y,x] = 1

                    pass

                last_mean = mean

        self.__show_scaled('vars', vars, 5)
        return []

    def __find_by_angle(self, angle: np.ndarray, magn: np.ndarray):

        angle_diff = lambda a1, a2: min((2 * np.pi) - abs(a1 - a2), abs(a1 - a2))

        abobus = np.zeros_like(angle)

        for y in range(1, angle.shape[0]):
            for x in range(1, angle.shape[1]):
                if magn[y,x] > self.detection_threshold:
                    abobus[y,x] = angle_diff(angle[y,x], angle[y, x-1])
        
        abobus /= np.pi
        self.__show_scaled('abobus', abobus, 5)

    def __inflateRectangle(self, frame: np.ndarray, start_y: int, start_x: int):
        '''Starting at (x,y), tries to find inner points of figure with BFS, and then fits it into rectangle'''
        rect = [start_x,start_y,start_x,start_y]

        visited = np.zeros_like(frame, dtype=bool)
        grid_height, grid_width = visited.shape

        queue = []

        visited[start_y, start_x] = True
        queue.append( (start_y, start_x) )

        steps = np.array(
            [[-1, 0], [1,  0], [0,  1], [0, -1],
             [-1,-1], [1,  1], [1, -1], [-1, 1]]
        )

        in_bounds = lambda y, x: 0 <= y < grid_height and 0 <= x < grid_width
        object_threshold = max(frame.max() * self.selection_threshold, 0.05)

        #self.__show_scaled('selection thresh', cv2.threshold(frame, object_threshold, 1, cv2.THRESH_BINARY)[1], 5)
        while queue:
            y, x = queue.pop(0)
            in_figure = False

            # part of object
            if frame[y, x] > object_threshold:
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
    
    def __insideContour(self, contours: list | np.ndarray, y: int, x: int):
        for rect in contours:
            if Geometry.inside((x,y), rect): 
                return True
        
        return False

    def __show_scaled(self, name: str, frame: np.ndarray, scale: int=5):
        cv2.imshow(
            name,
            cv2.resize(
                   frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_NEAREST
            )
        )