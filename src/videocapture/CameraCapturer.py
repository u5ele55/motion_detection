from videocapture.IVideoCapturer import IVideoCapturer

class CameraCapturer(IVideoCapturer):
    def __init__(self, index: int):
        super().__init__(index)