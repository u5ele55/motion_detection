import numpy as np

class IMotionDetector:
    '''Interface for motion detectors. '''
    def detect_motion(self, frame: np.ndarray, return_processed_frame=False):
        pass