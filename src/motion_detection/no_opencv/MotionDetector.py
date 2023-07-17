from motion_detection.IMotionDetector import *
from collections import deque
from utilities.geometry import *

class CVMotionDetector(IMotionDetector):
    def __init__(self, *, min_height: int=12, min_width: int=12, ):
        # We'll use mean of sequence of frames as value to which current frame is compared. 
        # Also we'll use binarization of their difference with big threshold, as noise couldn't 
        # be deleted with filtering
        pass

    def detect_motion(self, frame: np.ndarray, return_processed_frame: bool = False):
        pass