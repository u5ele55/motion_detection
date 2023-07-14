import cv2
from videocapture.CameraCapturer import CameraCapturer
from videocapture.VideofileCapturer import VideofileCapturer
from core.FrameDemonstration import FrameDemonstration
from motion_detection.opencv.MotionDetector import *

class Core:
    '''Class where all the components come together'''
    def start(self):
        video = VideofileCapturer(r'D:\Personal\Job\nic etu\Practic Tasks\static2.mp4')
        #video = CameraCapturer(0)
        original_window = FrameDemonstration('Original stream')
        grayscaled_window = FrameDemonstration('Grayscaled stream')

        ret, frame = video.next_frame()
        
        if not ret:
            raise RuntimeError("Cannot access video stream")
        
        height, width = frame.shape[:2]
        print("Video resolution: ", height, width)

        min_area = 24 * 32
        md = CVMotionDetector(height, width, min_area=min_area, threshold=5, capacity=5)

        while True:
            success, frame = video.next_frame()

            if success:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                contours, processed_frame = md.detect_motion(gray, return_processed_frame=True)
                for contour in contours:
                    x1,y1,x2,y2 = contour
                    cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)

                grayscaled_window.show_frame(processed_frame)
                original_window.show_frame(frame)
            else: 
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()