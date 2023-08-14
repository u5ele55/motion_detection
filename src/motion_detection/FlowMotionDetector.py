import numpy as np
from motion_detection.IMotionDetector import IMotionDetector
from utilities.geometry import Geometry
import cv2

from scipy.signal import medfilt2d

import warnings
warnings.filterwarnings("error")

MAX_FRAME_RESOLUTION_FLOW = 130_000

class FlowMotionDetector(IMotionDetector):
    def __init__(self, sample_frame: np.ndarray, *, 
                 grid_shape: tuple[int]=(7,7),
                 detection_threshold: float=0.3,
                 selection_threshold: float=2/10,
                 patience: int = 2,
                 max_elapsed_time: int = 5
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

        # To determine which objects stay in motion within few frames 
        self.patience = patience
        self.max_elapsed_time = max_elapsed_time

        self.figures = {}    # {rect_id: [vertices]}
        self.figure_center_streak = {} # {rect_id: [center_streak, elapsed_time, found_on_frame]}
        self.figure_cnt = 0

        

        
    def detect_motion(self, frame: np.ndarray, return_processed_frame=False):
        if self.new_size is not None: 
            frame = cv2.resize(frame, self.new_size)
        
        calc = cv2.calcOpticalFlowFarneback(self.previous, frame, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        mag, ang = cv2.cartToPolar(calc[:, :, 0], calc[:, :, 1])

        mag = medfilt2d(mag[::self.grid_shape[0], ::self.grid_shape[1]])
        ang = medfilt2d(ang[::self.grid_shape[0], ::self.grid_shape[1]])

        self.__show_scaled('magnitude filtered', mag / mag.max(), self.grid_shape[0])
        self.__show_scaled('angles filtered', ang / (2*np.pi), self.grid_shape[0])

        self.previous = frame

        magnitude_rect = self.__find_by_magnitude(mag)
        self.__find_by_angle(ang, mag)
        
        magnitude_rect = self.__process_figures(magnitude_rect, self.figures, self.figure_center_streak)

        if return_processed_frame:
            return magnitude_rect, ang
        return magnitude_rect
    
    def __find_by_magnitude(self, magn: np.ndarray):
        radius = 1
        changes = np.zeros_like(magn)
        for y in range(radius, magn.shape[0] - radius):
            for x in range(radius, magn.shape[1] - radius):
                mean = magn[y-radius:y+radius+1,x-radius:x+radius+1].mean() 
                changes[y,x] = abs(magn[y,x] - mean)  # maybe division? 
        
        self.__show_scaled('changes magnitude', changes, 5)
        
        changes_thresh = cv2.adaptiveThreshold((np.clip(changes, 0, 2)*127).astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -22)

        self.__show_scaled('thresh change in magnitude', changes_thresh, 5)
        
        rectangles = []

        for y in range(changes_thresh.shape[0]):
            for x in range(changes_thresh.shape[1]):
                if changes_thresh[y,x] == 255 and not self.__insideContour(rectangles, y, x):
                    rect = self.__inflate_thresholded(changes_thresh, magn, y,x)
                    if rect:
                        rectangles.append( rect )
        
        if not rectangles:
            return []

        rectangles = np.array(rectangles, dtype=float)

        rectangles[:, 0 :: 2] *= self.grid_shape[1] # scale to flow size
        rectangles[:, 1 :: 2] *= self.grid_shape[0]

        if self.new_size is not None:
            rectangles[:, 0 :: 2] *= self.old_size[0] / self.new_size[0] # scale to frame size
            rectangles[:, 1 :: 2] *= self.old_size[1] / self.new_size[1]

        return rectangles.astype('int32')

    def __find_by_angle(self, angle: np.ndarray, magn: np.ndarray):
        angle_diff = lambda a1, a2: min((2 * np.pi) - abs(a1 - a2), abs(a1 - a2))
        angle_grad = np.zeros_like(angle)
        
        for y in range(1, angle.shape[0]):
            for x in range(1, angle.shape[1]):
                if magn[y,x] > self.detection_threshold:
                    angle_grad[y,x] = angle_diff(angle[y,x], angle[y, x-1])
        

        angle_grad /= np.pi
        self.__show_scaled('X-angle difference', angle_grad, 5)
        grad_thresh = cv2.adaptiveThreshold((angle_grad*255).astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -25)
        self.__show_scaled('thresh grad magnitude', grad_thresh, 5)

        #changes_thresh = cv2.adaptiveThreshold((np.clip(magn, 0, 2)*127).astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -25)


    def __inflate_rectangle(self, frame: np.ndarray, start_y: int, start_x: int):
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
    
    def __inflate_thresholded(self, thresh: np.ndarray, original: np.ndarray, start_y: int, start_x: int):
        rect = [start_x,start_y,start_x,start_y]

        visited = np.zeros_like(thresh, dtype=bool)
        grid_height, grid_width = visited.shape

        queue = []

        visited[start_y, start_x] = True
        queue.append( (start_y, start_x) )

        steps = np.array(
            [[-1, 0], [1,  0], [0,  1], [0, -1],
             [-1,-1], [1,  1], [1, -1], [-1, 1]]
        )

        in_bounds = lambda y, x: 0 <= y < grid_height and 0 <= x < grid_width

        #self.__show_scaled('selection thresh', cv2.threshold(frame, object_threshold, 1, cv2.THRESH_BINARY)[1], 5)
        sum_coords = [0,0]
        sum_vals = 0
        cnt = 0


        while queue:
            y, x = queue.pop(0)
            in_figure = False

            # part of object
            if thresh[y, x] == 255:
                sum_coords[0] += y; sum_coords[1] += x
                sum_vals += original[y, x]
                cnt += 1

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
        
        
        # looking for a pivot
        est_y, est_x = round(sum_coords[0] / cnt), round(sum_coords[1] / cnt)
        if visited[est_y, est_x]:
            if sum_coords[0] < est_y * cnt:
                est_y -= 1
            elif sum_coords[0] > est_y * cnt:
                est_y += 1
            
            if visited[est_y, est_x]:
                if sum_coords[1] < est_x * cnt:
                    est_x -= 1
                elif sum_coords[1] > est_x * cnt:
                    est_x += 1
                
        # inflate from (est_y, est_x)
        queue.append( (est_y, est_x) )
        est_val = 255 if sum_vals / cnt > original.mean() else 0
        visited = np.zeros_like(thresh, dtype=bool)
        visited[est_y, est_x] = True

        original = cv2.adaptiveThreshold(
            (np.clip(original, 0, 2)*127).astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -22)
        self.__show_scaled('thresh magnitude', original, 5)

        while queue:
            y, x = queue.pop(0)
            in_figure = False

            # part of object
            if original[y, x] == est_val:
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

        if (rect[2] - rect[0]) * (rect[3] - rect[1]) >= 3 * visited.shape[0] * visited.shape[1] // 4 or \
            rect[2] - rect[0] == 0 or rect[3] - rect[1] == 0:
            return None

        return rect
    
    def __insideContour(self, contours: list | np.ndarray, y: int, x: int):
        for rect in contours:
            if Geometry.inside((x,y), rect): 
                return True
        
        return False

    def __process_figures(self, spotted: np.ndarray, aware: dict, aware_appearence: dict):
        used_rectangles = [False] * len(spotted)

        if not aware:
            for rect in spotted:
                aware[self.figure_cnt] = rect.copy()
                aware_appearence[self.figure_cnt] = [0, 0, True]
                self.figure_cnt += 1
        else:
            for i in range(len(spotted)):
                # find id s.t. aware[id] has center in spotted[i] borders (consider deviation) 
                for id in aware:
                    aware_appearence[id] = aware_appearence.get(id, [0, 0, False])
                    aware_appearence[id][2] = False
                    if Geometry.inside(Geometry.get_center(aware[id]), spotted[i]):
                        aware[id] = spotted[i].copy()
                        used_rectangles[i] = True
                        aware_appearence[id][0] += 1
                        aware_appearence[id][1] = 0
                        aware_appearence[id][2] = True
                        break
            # delete figures that wasn't found on frame
            to_del = []
            for id in aware_appearence:
                if not aware_appearence[id][2]:
                    # figure wasn't found
                    aware_appearence[id][1] += 1
                    if aware_appearence[id][1] >= self.max_elapsed_time:
                        to_del.append(id)
            for id in to_del:
                del aware[id]
                del aware_appearence[id]
        
        # iterate through non-used rectangles and add them to aware
        for i in range(len(spotted)):
            if not used_rectangles[i]:
                aware[self.figure_cnt] = spotted[i].copy()
                aware_appearence[self.figure_cnt] = [0, 0, True]
                self.figure_cnt += 1

        result = \
            [[i, aware[i]] for i in aware if aware_appearence[i][0] >= self.patience]
        
        return result

    def __show_scaled(self, name: str, frame: np.ndarray, scale: int=5):
        cv2.imshow(
            name,
            cv2.resize(
                   frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_NEAREST
            )
        )