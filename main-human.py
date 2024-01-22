import sys

from human.detection_copy import HumanDetection

if __name__ == '__main__':
    args = sys.argv[1]

    if args == "run":
        #s = HumanDetection(video_channel='sample/video/video-1.mp4', roi=(1, 1, 1280, 720))
        s = HumanDetection()
        s.detection()