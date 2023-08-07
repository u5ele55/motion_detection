import numpy as np
from motion_detection.IMotionDetector import IMotionDetector
import cv2

class FlowMotionDetector(IMotionDetector):
    def __init__(self):
        pass
    
    def detect_motion(self, frame: np.ndarray, return_processed_frame=False):

        return [], frame