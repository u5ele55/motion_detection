import numpy as np
from motion_detection.IMotionDetector import IMotionDetector
import cv2
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("error")

MAX_FRAME_RESOLUTION_FLOW = 130_000

class FlowMotionDetector(IMotionDetector):
    def __init__(self, sample_frame: np.ndarray, *, 
                 grid_shape: tuple[int]=(7,7),
                 training_sample_size: int=1000,
                 n_clusters: int=3
                 ):
        '''
        
        Parameters
        ----------
        grid_shape: tuple[int]
            step on x and y-axis when iterating through flow frame in search of figure
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
        size = self.new_size if self.new_size is not None else self.old_size
        self.clustering_step = size[0] * size[1] // training_sample_size

        self.kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
        self.frames_processed = 0

    
    def detect_motion(self, frame: np.ndarray, return_processed_frame=False):
        if self.new_size is not None: 
            frame = cv2.resize(frame, self.new_size)
        
        calc = cv2.calcOpticalFlowFarneback(self.previous, frame, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        mag, ang = cv2.cartToPolar(calc[:, :, 0], calc[:, :, 1])
        self.hsv[:, :, 0] = ang * 180 / np.pi / 2
        self.hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

        self.previous = frame
        
        clusterized_flow = self.__clusterize(flow)

        cv2.imshow('cluster', clusterized_flow)

        self.frames_processed += 1

        if return_processed_frame:
            return [], flow
        return []
    
    def __clusterize(self, flow: np.ndarray):
        flow = flow.astype(np.float64) / 255
        h,w,c = flow.shape
        assert c == 3, "Flow must be rgb... Smth isn't right..."
        flow_array = flow.reshape((w*h, c))

        if self.frames_processed % 10 == 0:
            flow_array_sample = flow_array[::self.clustering_step] # add regions where moving object was previously found
            print(len(flow_array_sample))
            self.kmeans.fit(flow_array_sample)
        labels = self.kmeans.predict(flow_array)

        return self.kmeans.cluster_centers_[labels].reshape(h, w, -1)
        
    def __color_diff(self, c1, c2):
        ''' colors in bgr '''
        c1 = c1.astype('int32')
        c2 = c2.astype('int32')
        rmean = ( c1[2] + c2[2] ) / 2

        b = c1[0] - c2[0]
        g = c1[1] - c2[1]
        r = c1[2] - c2[2]
        
        return np.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256))
    
    def __inflateRectangle(self, frame, start_y, start_x):
        '''Starting at (x,y), tries to find inner points of figure with BFS, and then fits it into rectangle'''
        rect = [start_x,start_y,start_x,start_y]

        visited = np.zeros_like(frame, dtype=bool)
        grid_height, grid_width = visited.shape

        queue = []

        visited[start_y, start_x] = True
        queue.append( (start_y, start_x) )

        steps = np.array(
            [[-1,  0], [1,  0], [0,  1], [0,  -1],
             [-1, -1], [1,  1], [1, -1], [-1,  1]]
        )

        in_bounds = lambda y, x: 0 <= y < grid_height and 0 <= x < grid_width
        self.object_threshold = frame.max() * 1/10 + frame.min() * 9/10
        while queue:
            y, x = queue.pop(0)
            in_figure = False

            # part of object
            if frame[y, x] > self.object_threshold:
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