import numpy as np

class IMotionDetector:
    '''Interface for motion detectors. '''
    def detect_motion(self, frame: np.ndarray) -> np.ndarray:
        '''Returns numpy array (N, 4) for N detected objects and its rectangle vertices.
        \n `frame` should be grayscaled'''
        pass