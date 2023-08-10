import cv2
from videocapture.CameraCapturer import CameraCapturer
from videocapture.VideofileCapturer import VideofileCapturer
from visual.FrameDemonstration import FrameDemonstration
from visual.TrajectoryDrawer import TrajectoryDrawer
from motion_detection.FlowMotionDetector import FlowMotionDetector

class Core:
    '''Class where all the components come together'''
    def start(self):
        video = VideofileCapturer(r'C:\Users\vshaganov\workplace\tests\trees.mp4')
        video = VideofileCapturer(r'D:\Personal\Job\nic etu\Practic Tasks\drone_traffic.mp4')

        original_window = FrameDemonstration('Original stream')
        grayscaled_window = FrameDemonstration('Changes')

        ret, frame = video.next_frame()
        
        if not ret:
            raise RuntimeError("Cannot access video stream")
        
        height, width = frame.shape[:2]
        print("Video resolution: ", height, width)

        md = FlowMotionDetector(frame)

        while True:
            success, frame = video.next_frame()

            if success:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                contours, processed_frame = md.detect_motion(gray, return_processed_frame=True)
                # draw contours
                

                grayscaled_window.show_frame(processed_frame)
                original_window.show_frame(frame)
            else: 
                break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            

        cv2.destroyAllWindows()