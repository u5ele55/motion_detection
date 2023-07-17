from motion_detection.IMotionDetector import *
from collections import deque
from utilities.geometry import *

class CVMotionDetector(IMotionDetector):
    def __init__(self, *, min_height: int=12, min_width: int=12, ):
        pass

    def detect_motion(self, frame: np.ndarray, return_processed_frame: bool = False):
        pass