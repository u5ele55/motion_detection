from motion_detection.IMotionDetector import *
import cv2
from collections import deque
from utilities.geometry import *

class CVMotionDetector(IMotionDetector):
    def __init__(self, height: int, width: int, *, capacity: int = 10, min_area: int = 50, dilation: int = 7, threshold: int = 15):
        self.frame_pool = deque([]) 
        self.pool_sum = np.zeros((height, width))
        self.min_area = min_area
        self.capacity = capacity
        self.dilation = dilation
        self.threshold = threshold
    
    def detect_motion(self, frame: np.ndarray, return_processed_frame: bool = False) -> np.ndarray:
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
            for j in range(i+1, len(result)):
                if mask[j] and Geometry.contains(result[i], result[j]):
                    mask[j] = False
        result = np.array([result[i] for i in range(len(result)) if mask[i]])
        if return_processed_frame:
            return result, thresh_frame
        return result