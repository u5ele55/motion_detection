import numpy as np

class Geometry:
    @staticmethod
    def contains(big_rect: np.ndarray, small_rect: np.ndarray) -> bool:
        '''Checks if `big_rect` contains `small_rect` (not just intersects) 
        \n rectangles are of format [x1, y1, x2, y2]'''
        
        X1,Y1, X2,Y2 = big_rect
        x1,y1, x2,y2 = small_rect

        return X1 <= x1 and x2 <= X2 and Y1 <= y1 and y2 <= Y2
    @staticmethod
    def inside(point: list | np.ndarray, rect: np.ndarray) -> bool:
        '''Checks if `point` is inside rectangle `rect`'''
        x, y = point
        x1,y1, x2,y2 = rect

        return x1 <= x <= x2 and y1 <= y <= y2