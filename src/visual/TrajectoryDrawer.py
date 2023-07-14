import cv2
from utilities.geometry import Geometry

class TrajectoryDrawer:
    def __init__(self, memory_size: int = 20):
        self.memory_size = memory_size
        self.history = {} # {id: queue[...last N positions...]}
    
    def draw(self, frame, figures):
        '''Draws trajectory of figures according to their positions in frames.
        Parameters
        ----------
        frame: np.ndarray
            frame itself
        figures: list
            list of format [ [id, [vertices]], ... ]'''
        for id in figures:

            #self.history[id](Geometry.get_center(self.history[id]))
        for i in self.history:

        return