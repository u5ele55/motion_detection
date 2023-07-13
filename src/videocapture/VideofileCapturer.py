from videocapture.IVideoCapturer import IVideoCapturer

class VideofileCapturer(IVideoCapturer):
    def __init__(self, filename: str):
        super().__init__(filename)