import cv2
from videocapture.CameraCapturer import CameraCapturer
from core.FrameDemonstration import FrameDemonstration
from motion_detection.opencv.MotionDetector import *

class Core:
    '''Class where all the components come together'''
    def start(self):
        camera = CameraCapturer(0)
        original_window = FrameDemonstration('Original stream')
        grayscaled_window = FrameDemonstration('Grayscaled stream')

        ret, frame = camera.next_frame()
        md = CVMotionDetector(frame.shape[0], frame.shape[1], min_area=100)

        while True:
            ret, frame = camera.next_frame()

            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                contours, fr = md.detect_motion(gray, return_processed_frame=True)
                for contour in contours:
                    x1,y1,x2,y2 = contour
                    cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)

                grayscaled_window.show_frame(fr)
                original_window.show_frame(frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()