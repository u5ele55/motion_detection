import cv2
from videocapture.CameraCapturer import CameraCapturer
from videocapture.VideofileCapturer import VideofileCapturer
from visual.FrameDemonstration import FrameDemonstration
from visual.TrajectoryDrawer import TrajectoryDrawer
from motion_detection.opencv.MotionDetector import *


import time

class Core:
    '''Class where all the components come together'''
    def start(self):
        video = VideofileCapturer(r'C:\Users\vshaganov\workplace\tests\light_traffic.mp4')
        #video = CameraCapturer(0)
        original_window = FrameDemonstration('Original stream')
        grayscaled_window = FrameDemonstration('Grayscaled stream')

        ret, frame = video.next_frame()
        
        if not ret:
            raise RuntimeError("Cannot access video stream")
        
        height, width = frame.shape[:2]
        print("Video resolution: ", height, width)

        min_area = 24 * 32
        md = CVMotionDetector(height, width, min_area=min_area, threshold=5, capacity=5, max_elapsed_time=5)

        gen_color = lambda id: (id * 219 % 255, id * 179 % 255, id * 301 % 255)
        trajectory = TrajectoryDrawer(color_generator=gen_color, memory_size=120)

        while True:
            success, frame = video.next_frame()

            if success:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                contours, processed_frame = md.detect_motion(gray, return_processed_frame=True)
                for ind, contour in contours:
                    x1,y1,x2,y2 = contour
                    cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(x2, y2), color=gen_color(ind), thickness=2)
                    
                frame = trajectory.draw(frame, contours)
                grayscaled_window.show_frame(processed_frame)
                original_window.show_frame(frame)
            else: 
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            

        cv2.destroyAllWindows()