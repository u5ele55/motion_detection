import cv2
from videocapture.CameraCapturer import CameraCapturer
from videocapture.VideofileCapturer import VideofileCapturer
from visual.FrameDemonstration import FrameDemonstration
from visual.TrajectoryDrawer import TrajectoryDrawer
from motion_detection.opencv.MotionDetector import *

from motion_detection.no_opencv.MotionDetector import CustomMotionDetector

import time

class Core:
    '''Class where all the components come together'''
    def start(self):
        video = VideofileCapturer(r'C:\Users\vshaganov\workplace\tests\topview.mp4')
        #video = VideofileCapturer(r'C:\Users\vshaganov\workplace\tests\clouds_2.mp4')

        original_window = FrameDemonstration('Original stream')
        grayscaled_window = FrameDemonstration('Changes')

        ret, frame = video.next_frame()
        
        if not ret:
            raise RuntimeError("Cannot access video stream")
        
        height, width = frame.shape[:2]
        print("Video resolution: ", height, width)

        min_area = 12 * 16

        md = CVMotionDetector(height, width, min_area=min_area, threshold=10, 
                              capacity=15, max_elapsed_time=5, patience=5)
        '''md = CustomMotionDetector(capacity=3, object_threshold=3, move_threshold=10, 
                                  patience=10, max_elapsed_time=10, min_area=min_area,
                                  detection_step=(16,16), object_selection_step=(5,5))
        '''
        gen_color = lambda id: (id * 219 % 255, id * 179 % 255, id * 301 % 255)
        trajectory = TrajectoryDrawer(color_generator=gen_color, memory_size=40)

        while True:
            success, frame = video.next_frame()

            if success:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                contours, processed_frame = md.detect_motion(gray, return_processed_frame=True)
                for ind, contour in contours:
                    x1,y1,x2,y2 = contour[:4]
                    cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(x2, y2), color=gen_color(ind), thickness=2)
                    
                frame = trajectory.draw(frame, contours)
                grayscaled_window.show_frame(processed_frame)
                original_window.show_frame(frame)
            else: 
                break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            

        cv2.destroyAllWindows()