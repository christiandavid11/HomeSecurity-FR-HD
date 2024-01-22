import asyncio
import sys
import os
import shutil
import random
import string

async def main(args: str):

    if args == "run":
        # from face.l_recog import FaceRecognition
        # print("after importing face")
        dir = "/home/pi/Desktop/human-detection-main"
        fname = "passcode.txt"

        if os.path.isfile(os.path.join(dir, fname)):
            print(f"{fname} exists in {dir}")
        else:
            from authentication import validate_pin
            validate_pin(None)


        from human.final import HumanDetection
        print("after importing human")

 
        
        await asyncio.create_task(HumanDetection().detection())
        # await asyncio.create_task(FaceRecognition().face_recognize())

        # from human.n_detect import HumanDetection
        # await asyncio.create_task(HumanDetection().detection())

    elif args == "train":
        from face.encoder import EncodeFaces
        from modules import get_images

        dir = "/home/pi/Desktop/human-detection-main"
        fname = "productkey.txt"

        if os.path.isfile(os.path.join(dir, fname)):
            print(f"{fname} exists in {dir}")
        else:
            print(f"{fname} does not exist in {dir}")
            random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
            formatted_string = '-'.join([random_string[i:i+4].upper() for i in range(0, len(random_string), 4)])
            
            fpath = os.path.join(dir, fname)

            with open(fpath, "w") as file:
                file.write("Product Code: " + formatted_string)
            print(f"{fname} has been created and saved in {dir}.")

        


        ################################################################
        #await create_env()
        directory = '/home/pi/Desktop/human-detection-main/src/dataset'

        for filename in os.listdir(directory):
            ndir = '/home/pi/Desktop/human-detection-main/src/dataset/' + filename
            try:
                shutil.rmtree(ndir)
                print(f'Deleted: {ndir}')
            except Exception as e:
                print('Error: {e}')

        get_images()
        ef = EncodeFaces()
        ef.encode_faces()

    else:
        print('[ERROR] Wrong paramater passed...')

if __name__ == "__main__":

    args = sys.argv[1]
    asyncio.run(main(args))

#source ~/env/myenv/bin/actvate