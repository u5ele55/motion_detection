from motion_detection.IMotionDetector import *
import cv2
from collections import deque
from utilities.geometry import *

class CVMotionDetector(IMotionDetector):
    def __init__(self, height: int, width: int, *, capacity: int = 10, min_area: int = 50, dilation: int = 7, threshold: int = 15, max_deviation: float = 0):
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
            maximal deviation of the center of an object relatively to its rectangle borders compared to previous frame. 
            initialize as `-1` to disable this feature.
        '''
        self.frame_pool = deque([]) 
        self.pool_sum = np.zeros((height, width))

        self.min_area  = min_area
        self.capacity  = capacity
        self.dilation  = dilation
        self.threshold = threshold
        self.centers   = []
        self.max_deviation = max_deviation
    
    def detect_motion(self, frame: np.ndarray, return_processed_frame: bool = False):
        frame = cv2.GaussianBlur(frame, ksize=(5,5), sigmaX=0)

        if len(self.frame_pool) != self.capacity:
            self.frame_pool.append(frame.copy())
            self.pool_sum += frame
            if return_processed_frame:
                return np.array([]), self.pool_sum / len(self.frame_pool)
            return np.array([])
        
        diff = cv2.absdiff(frame, (self.pool_sum / self.capacity).astype('uint8'))
        diff = cv2.dilate(diff, np.ones((self.dilation, self.dilation)))
        thresh_frame = cv2.threshold(src=diff, thresh=self.threshold, maxval=255, type=cv2.THRESH_BINARY)[1]
        
        # Updating mean
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

        for i in range(len(result)):
            if not mask[i]: 
                continue
            for j in range(i+1, len(result)):
                if mask[j] and Geometry.contains(result[i], result[j]):
                    mask[j] = False
        result = np.array([result[i] for i in range(len(result)) if mask[i]])
        if self.centers:
            # check if centers lying inside rectangles from `result` 
            a = 1
        self.centers = [((rect[0] + rect[2])/2, (rect[1] + rect[3])/2) for rect in result] 
        
        if return_processed_frame:
            return result, thresh_frame
        return result