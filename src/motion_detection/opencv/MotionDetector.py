from motion_detection.IMotionDetector import *
import cv2
from collections import deque
from utilities.geometry import *

class CVMotionDetector(IMotionDetector):
    def __init__(self, 
                 height: int, 
                 width: int, *, 
                 capacity: int = 10, 
                 min_area: int = 50, 
                 dilation: int = 7, 
                 threshold: int = 15, 
                 max_deviation: int = 0, 
                 patience: int = 2,
                 max_elapsed_time: int = 5):
        '''

        Parameters 
        ----------
        height : int
            height of a frame
        width : int
            width of a frame
        capacity : int
            quantity of previous frames which are to used detect motion on a new frame
        min_area : int
            minimal area of an object to consider
        dilation : int
            dilation kernel size that used on frame difference
        threshold : int
            minimal brightness of object in frame difference to consider
        max_deviation : int
            maximal deviation of the center of an object relatively to its rectangle borders (in L1 norm) compared to previous frame. 
            initialize as `-1` to disable this feature.
        patience : int
            quantity of frames to wait to admit figure presence on frame
        max_elapsed_time : int
            quantity of frames to wait for previously moving object to admit its stopped 
        '''
        self.frame_pool = deque([]) 
        self.pool_sum = np.zeros((height, width))

        self.figures = {}    # {rect_id: [vertices]}
        self.figure_center_streak = {} # {rect_id: [center_streak, elapsed_time, found_on_frame]}
        self.figure_cnt = 0

        self.min_area  = min_area
        self.capacity  = capacity
        self.dilation  = dilation
        self.threshold = threshold
        self.max_deviation = max_deviation
        self.patience = patience
        self.max_elapsed_time = max_elapsed_time
    
    def detect_motion(self, frame: np.ndarray, return_processed_frame: bool = False):
        frame = cv2.GaussianBlur(frame, ksize=(7,7), sigmaX=0)

        # store first `capacity` frames and accumulate their sum
        if len(self.frame_pool) != self.capacity:
            self.frame_pool.append(frame.copy())
            self.pool_sum += frame
            if return_processed_frame:
                return np.array([]), self.pool_sum / len(self.frame_pool)
            return np.array([])
        
        diff = cv2.absdiff(frame, (self.pool_sum / self.capacity).astype('uint8'))
        diff = cv2.dilate(diff, np.ones((self.dilation, self.dilation)))
        thresh_frame = cv2.threshold(src=diff, thresh=self.threshold, maxval=255, type=cv2.THRESH_BINARY)[1]
        
        # Update mean
        self.pool_sum -= self.frame_pool.popleft()
        self.frame_pool.append(frame.copy())
        self.pool_sum += frame
        
        contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        result = []

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            result.append(np.array([x, y, x+w, y+h]))

        # sort by area
        result = sorted(result, key=lambda rect: (rect[2]-rect[0])*(rect[3]-rect[1]), reverse=True)
        mask = [True] * len(result)

        # delete inner rectangles
        for i in range(len(result)):
            if not mask[i]: 
                continue
            for j in range(i+1, len(result)):
                if mask[j] and Geometry.contains(result[i], result[j]):
                    mask[j] = False
        # delete those that were marked as inner 
        result = [result[i] for i in range(len(result)) if mask[i]]

        # take only rectangles for which there was center on previous frames
        get_center = lambda r: ((r[0] + r[2])/2, (r[1] + r[3])/2)
        used_rectangles = [False] * len(result)

        if not self.figures:
            for rect in result:
                self.figures[self.figure_cnt] = rect.copy()
                self.figure_center_streak[self.figure_cnt] = [0, 0, True]
                self.figure_cnt += 1
        elif self.max_deviation >= 0:
            for i in range(len(result)):
                inflated_rectangle = [
                    result[i][0]-self.max_deviation, result[i][1]-self.max_deviation,
                    result[i][2]+self.max_deviation, result[i][3]+self.max_deviation,
                    ]
                # find id s.t. self.figures[id] has center in result[i] borders (consider deviation) 
                for id in self.figures:
                    self.figure_center_streak[id] = self.figure_center_streak.get(id, [0, 0, False])
                    self.figure_center_streak[id][2] = False
                    if Geometry.inside(get_center(self.figures[id]), inflated_rectangle):
                        self.figures[id] = result[i].copy()
                        used_rectangles[i] = True
                        self.figure_center_streak[id][0] += 1
                        self.figure_center_streak[id][1] = max(0, self.figure_center_streak[id][1]-1)
                        self.figure_center_streak[id][2] = False
            # delete figures that wasn't found on frame
            to_del = []
            for id in self.figure_center_streak:
                if self.figure_center_streak[id][2]:
                    # figure wasn't found
                    self.figure_center_streak[id][1] += 1
                    if self.figure_center_streak[id][1] >= self.max_elapsed_time:
                        to_del.append(id)
            for id in to_del:
                del self.figures[id]
                del self.figure_center_streak[id]
        
        # iterate through non-used rectangles and add them to self.figures
        for i in range(len(result)):
            if not used_rectangles[i]:
                self.figures[self.figure_cnt] = result[i].copy()
                self.figure_center_streak[self.figure_cnt] = [0, 0, True]
                self.figure_cnt += 1
        
        print(self.figure_center_streak)

        result = np.array(
            [self.figures[i] for i in self.figures if self.figure_center_streak[i][0] >= self.patience])
        print(result)
        if return_processed_frame:
            return result, thresh_frame
        return result