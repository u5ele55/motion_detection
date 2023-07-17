import cv2
from utilities.geometry import Geometry
from collections import deque
import numpy as np

class TrajectoryDrawer:
    def __init__(self, memory_size: int = 40, color_generator=lambda x: (0,255,0)):
        self.memory_size = memory_size
        self.history = {} # {id: [...last N positions...]}
        self.frame_counter = {}
        self.color_generator = color_generator
    
    def draw(self, frame, figures):
        '''Draws trajectory of figures according to their positions in frames.
        Parameters
        ----------
        frame: np.ndarray
            frame itself
        figures: list
            list of format [ [id, [vertices]], ... ]
        '''
        

        for id, figure in figures:
            self.history.setdefault(id, np.array(Geometry.get_center(figure), ndmin=2))
            self.history[id] = np.append(self.history[id], [Geometry.get_center(figure)], axis=0)

        to_del = []
        for id in self.history:
            for i in range(len(self.history[id]) - 1):
                frame = cv2.line(frame, self.history[id][i], self.history[id][i+1], color=self.color_generator(id), thickness=(self.memory_size+4*i+1)//self.memory_size)
            self.frame_counter[id] = self.frame_counter.get(id, 0) + 1
            if self.frame_counter[id] >= self.memory_size:
                self.history[id] = self.history[id][1:]
        
                if not len(self.history[id]):
                    to_del.append(id)

        for id in to_del:    
            del self.history[id]
            del self.frame_counter[id]
        
        return frame