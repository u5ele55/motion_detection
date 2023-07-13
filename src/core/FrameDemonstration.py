import cv2

class FrameDemonstration:
    def __init__(self, window_title: str):
        self.window_title = window_title
    def show_frame(self, frame):
        cv2.imshow(self.window_title, frame)