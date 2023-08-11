import numpy as np
from motion_detection.IMotionDetector import IMotionDetector
from utilities.geometry import Geometry
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
                 n_clusters: int=3,
                 detection_threshold: float=5/6,
                 selection_threshold: float=2/10
                 ):
        '''
        
        Parameters
        ----------
        grid_shape: tuple[int]
            step on y and x-axis when iterating through flow frame in search of figure
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
        assert 0 <= detection_threshold <= 1 and 0 <= selection_threshold <= 1 
        self.detection_threshold = detection_threshold
        self.selection_threshold = selection_threshold
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

        figures = self.__find_figures(clusterized_flow)

        self.frames_processed += 1
        
        if return_processed_frame:
            return figures, flow
        return figures
    
    def __clusterize(self, flow: np.ndarray):
        h,w,c = flow.shape
        assert c == 3, "Flow must be rgb... Smth isn't right..."
        flow_array = (flow.astype(np.float64) / 255).reshape((w*h, c))

        if self.frames_processed % 10 == 0:
            flow_array_sample = flow_array[::self.clustering_step] # add regions where moving object was previously found
            self.kmeans.fit(flow_array_sample)

        labels = self.kmeans.predict(flow_array)
        
        return self.kmeans.cluster_centers_[labels].reshape(h, w, -1)

    def __find_figures(self, image: np.ndarray):
        grad = self.__find_gradient(image)
        cv2.imshow('grad', 
                cv2.resize(grad, (grad.shape[1] * 5, grad.shape[0] * 5), interpolation=cv2.INTER_NEAREST)
        )
        thresh = max(grad.max() * self.detection_threshold, 0.2)
        self.__show_scaled('grad threshold', cv2.threshold(grad, thresh, 1, cv2.THRESH_BINARY)[1], 5)

        rects = []
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                if grad[i,j] > thresh:
                    if self.__insideContour(rects, i, j):
                        continue
                    rects.append( self.__inflateRectangle(grad, i,j) )
        rects = np.array(rects, dtype=float)
        rects[:, 0 :: 2] *= self.grid_shape[1] # scale to flow size
        rects[:, 1 :: 2] *= self.grid_shape[0]

        if self.new_size is not None:
            rects[:, 0 :: 2] *= self.old_size[0] / self.new_size[0] # scale to frame size
            rects[:, 1 :: 2] *= self.old_size[1] / self.new_size[1]
        
        return rects.astype("int32")


    def __find_gradient(self, image: np.ndarray):
        ''' image is clusterized '''

        image *= 255
        gradX = np.zeros((image.shape[0] // self.grid_shape[0], image.shape[1] // self.grid_shape[1]))

        for i in range(0, gradX.shape[0]):
            for j in range(0, gradX.shape[1]):
                gradX[i,j] = self.__color_diff(
                    image[i * self.grid_shape[0], j * self.grid_shape[1]], 
                    image[i * self.grid_shape[0], (j-1) * self.grid_shape[1]] )
        
        '''cv2.imshow('gradX', 
            cv2.threshold(
                cv2.resize(gradX, (gradX.shape[1] * 5, gradX.shape[0] * 5), interpolation=cv2.INTER_NEAREST),
                0.3, 1, cv2.THRESH_BINARY
            )[1]
        )'''

        gradY = np.zeros((image.shape[0] // self.grid_shape[0], image.shape[1] // self.grid_shape[1]))

        for i in range(0, gradY.shape[0]):
            for j in range(0, gradY.shape[1]):
                gradY[i,j] = self.__color_diff(
                    image[i * self.grid_shape[0], j * self.grid_shape[1]], 
                    image[(i-1) * self.grid_shape[0], j * self.grid_shape[1]] )
        
        '''cv2.imshow('gradY', 
            cv2.threshold(
                cv2.resize(gradY, (gradY.shape[1] * 5, gradY.shape[0] * 5), interpolation=cv2.INTER_NEAREST),
                0.3, 1, cv2.THRESH_BINARY
            )[1]
        )'''

        gradsum = (gradY+gradX) / (2*768)

        return gradsum
        
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
            [[-1, 0], [1,  0], [0,  1], [0, -1],
             [-1,-1], [1,  1], [1, -1], [-1, 1]]
        )

        in_bounds = lambda y, x: 0 <= y < grid_height and 0 <= x < grid_width
        object_threshold = max(frame.max() * self.selection_threshold, 0.05)

        self.__show_scaled('selection thresh', cv2.threshold(frame, object_threshold, 1, cv2.THRESH_BINARY)[1], 5)
        while queue:
            y, x = queue.pop(0)
            in_figure = False

            # part of object
            if frame[y, x] > object_threshold:
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
    
    def __insideContour(self, contours, y, x):
        for rect in contours:
            if Geometry.inside((x,y), rect): 
                return True
        
        return False

    def __show_scaled(self, name, frame, scale=5):
        cv2.imshow(
            name,
            cv2.resize(
                   frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_NEAREST
            )
        )