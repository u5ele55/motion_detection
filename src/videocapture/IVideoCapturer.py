import cv2

class IVideoCapturer:
    def __init__(self, *args, **kwargs):
        self.stream = cv2.VideoCapture(*args, **kwargs)
    def next_frame(self):
        '''Returns next frame from video stream'''
        ret, frame = self.stream.read()
        return ret, frame
    def __del__(self):
        self.stream.release()
        print(f"Video stream released")