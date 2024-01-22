import asyncio
import sys


async def main(args: str):

    if args == "run":
        from face.facerecog import FaceRecognition
        from human.humandetect import HumanDetection

        await asyncio.create_task(FaceRecognition().face_recognize())
        await asyncio.create_task(HumanDetection().detection())

    elif args == "train":
        from face.encoder import EncodeFaces
        from modules import create_env, get_images

        #await create_env()
        await get_images()
        ef = EncodeFaces()
        ef.encode_faces()

    else:
        print('[ERROR] Wrong paramater passed...')

if __name__ == "__main__":

    args = sys.argv[1]
    asyncio.run(main(args))