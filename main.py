import sys
import asyncio

if __name__ == "__main__":
    args = sys.argv[1]

    if args == "run":
        #from human.detection import HumanDetection
        from face.recognizer import FaceRecognition

        #hd = HumanDetection(video_channel=1)
        fr = FaceRecognition(video_channel=0)

        #asyncio.run(hd.detection())
        asyncio.run(fr.face_recognize())

    elif args == "train":
        from face.encoder import EncodeFaces
        from modules import get_images

        #create_env()
        get_images()
        ef = EncodeFaces()
        ef.encode_faces()

    else:
        print('[ERROR] Wrong paramater passed...')
