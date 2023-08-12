import cv2
from videocapture.CameraCapturer import CameraCapturer
from videocapture.VideofileCapturer import VideofileCapturer
from visual.FrameDemonstration import FrameDemonstration
from visual.TrajectoryDrawer import TrajectoryDrawer
from motion_detection.FlowMotionDetector import FlowMotionDetector

class Core:
    '''Class where all the components come together'''
    def start(self):
        #video = VideofileCapturer(r'C:\Users\vshaganov\workplace\tests\trees.mp4')
        video = VideofileCapturer(r'D:\Personal\Job\nic etu\Practic Tasks\drone_top.mp4')

        original_window = FrameDemonstration('Original stream')
        grayscaled_window = FrameDemonstration('Changes')

        ret, frame = video.next_frame()
        
        if not ret:
            raise RuntimeError("Cannot access video stream")
        
        height, width = frame.shape[:2]
        print("Video resolution: ", height, width)

        md = FlowMotionDetector(frame, selection_threshold=3/10, detection_threshold=0.3)

        while True:
            success, frame = video.next_frame()

            if success:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                contours = md.detect_motion(gray, return_processed_frame=False)
                # draw contours
                for contour in contours:
                    x1,y1,x2,y2 = contour[:4]
                    cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)

                #grayscaled_window.show_frame(processed_frame)
                original_window.show_frame(frame)
            else: 
                break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            

        cv2.destroyAllWindows()