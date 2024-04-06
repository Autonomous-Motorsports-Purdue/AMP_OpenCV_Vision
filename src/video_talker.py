import cv2
import rpyc

class VideoTalker(rpyc.Service):
    def __init__(self):
        super().__init__()
        self.video = Video()
    def on_connect(self, conn):
        print("Talker connected")
        return "Talker connected"
    def on_disconnect(self, conn):
        print("Talker disconnected")
        return "Talker disconnected"
    def exposed_get_image(self):
        return self.video.get_video_data()
class Video:
    def __init__(self):
        self.vid = cv2.VideoCapture(0)
    def get_video_data(self):
        ret, frame = self.vid.read()
        cv2.imshow('frame', frame)
        return frame
        


if __name__ == '__main__':
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(VideoTalker, port = 9001)
    t.start()