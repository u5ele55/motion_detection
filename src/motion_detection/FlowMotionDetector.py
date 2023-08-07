import numpy as np
from motion_detection.IMotionDetector import IMotionDetector
import cv2

import warnings
warnings.filterwarnings("error")

MAX_FRAME_RESOLUTION_FLOW = 130_000

class FlowMotionDetector(IMotionDetector):
    def __init__(self, sample_frame: np.ndarray, patience: int=2):
        self.new_size = None

        H, W = sample_frame.shape[:2]

        if H * W > MAX_FRAME_RESOLUTION_FLOW:
            # H * W < MAX_FRAME_RESOLUTION_FLOW
            # H * W * a^2 < MAX_FRAME_RESOLUTION_FLOW
            # a^2 = MAX_FRAME_RESOLUTION_FLOW / H*W
            a = np.sqrt(MAX_FRAME_RESOLUTION_FLOW / (H * W))
            new_w, new_h = int(W * a), int(H * a)
            self.new_size = (new_w, new_h)
            sample_frame = cv2.resize(sample_frame, self.new_size)

        self.previous = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
        self.hsv = np.zeros_like(sample_frame)
        self.hsv[:, :, 1] = 255
    
    def detect_motion(self, frame: np.ndarray, return_processed_frame=False):
        if self.new_size is not None: 
            frame = cv2.resize(frame, self.new_size)
        
        next = cv2.calcOpticalFlowFarneback(self.previous, frame, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        mag, ang = cv2.cartToPolar(next[:, :, 0], next[:, :, 1])
        self.hsv[:, :, 0] = ang * 180 / np.pi / 2
        self.hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

        self.previous = frame
        
        # post-proceessing here (extract figures from flow)
        figures = self.__extract_figures(flow)

        if return_processed_frame:
            return [], flow
        return []
    
    def __extract_figures(self, flow: np.ndarray):
        H, W = flow.shape[:2]
        
        # perform grid checking looking for pixels that arent close to estimated pixel (which could be calculated as mean of previous pixels or mean of some area around current pixel)

    def __color_diff(self, c1, c2):
        ''' colors in bgr '''
        c1 = c1.astype('int32')
        c2 = c2.astype('int32')
        rmean = ( c1[2] + c2[2] ) / 2

        try:
            b = c1[0] - c2[0]
            g = c1[1] - c2[1]
            r = c1[2] - c2[2]
        except:
            print(c1, c2)
        return np.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256))