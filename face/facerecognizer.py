import time
import face_recognition
import pickle
import cv2
from datetime import datetime
import numpy as np
from krakenio import Client

from modules import detected, is_ready

class FaceRecognition:

    encodings='encodings.pickle'
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    data = pickle.loads(open(encodings, "rb").read())
    recognize = 'output/authorize'
    unrecognize = 'output/unauthorize'

    def __init__(self, video_channel=0, output='output/video.avi', detection_method='hog'):
        self.output = output
        self.video_channel = video_channel
        self.detection_method = detection_method
        self.authorize_output = 'output/authorize'
        self.unauthorize_output = 'output/unauthorize'

        is_ready("face-recognized", True)


    async def face_recognize(self):
        cap = cv2.VideoCapture(self.video_channel)
        writer = None

        while True:
            ret, frame = cap.read()
            color = (0, 255, 0)

            if ret is False:
                print('[ERROR] Something wrong with your camera...')
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r = frame.shape[1] / float(rgb.shape[1])

            boxes = face_recognition.face_locations(rgb, model=self.detection_method)
            encodings = face_recognition.face_encodings(rgb, boxes)
            
            names = []
            
            if boxes and encodings:
                for encoding in encodings:
                    matches = face_recognition.compare_faces(self.data["encodings"], encoding)
                    name = "Unknown"

                    if True in matches:
                        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                        counts = {}

                        for i in matchedIdxs:
                            name = self.data["names"][i]
                            counts[name] = counts.get(name, 0) + 1

                        name = max(counts, key=counts.get)

                    names.append(name)

                for ((top, right, bottom, left), name) in zip(boxes, names):
                    top = int(top * r)
                    right = int(right * r)
                    bottom = int(bottom * r)
                    left = int(left * r)

                    if name == "Unknown":
                        color = (0, 0, 255)
                        photo = f'{self.unauthorize_output}/{datetime.now().strftime("%d_%m_%Y_%H_%M")}_{name}.jpg'
                        cv2.imwrite(photo, frame)
                        api = Client('945b1b651bb94cbde60be627bbaeacb3', 'd5c230f96920f767cb3dadfb4faf534dae1eaea7')
                        data = {
                            'wait': True
                        }

                        result = api.upload(photo, data)
                        img_url = ""
                        if result.get('success'):
                            img_url = result.get('kraked_url')
                            print (img_url)
                        else:
                            print (result.get('message'))

                        await detected(type="face-recognized", is_detected=False, name=name, uploaded_file=img_url)

                    else:
                        
                        temp_name = f'{datetime.now().strftime("%d_%m_%Y_%H_%M")}_{name}.jpg'
                        #print(temp_name)
                        photo = f'{self.authorize_output}/{datetime.now().strftime("%d_%m_%Y_%H_%M")}_{name}.jpg'
                        cv2.imwrite(photo, frame)

                        api = Client('945b1b651bb94cbde60be627bbaeacb3', 'd5c230f96920f767cb3dadfb4faf534dae1eaea7')
                        data = {
                            'wait': True
                        }

                        result = api.upload(photo, data)
                        img_url = ""
                        if result.get('success'):
                            img_url = result.get('kraked_url')
                            print (img_url)
                        else:
                            print (result.get('message'))

                        await detected(type="face-recognized", is_detected=True, name=name, uploaded_file=img_url)
 
                    # - Start -
                    # if name == "Unknown":
                    #     detected("face-recognized", False, name)
                    #     color = (0, 0, 255)
                    #     cv2.imwrite(
                    #         f'{self.unauthorize_output}/{datetime.now().strftime("%d_%m_%Y_%H_%M")}_{name}.jpg',
                    #         frame
                    #     )
                    # else:
                    #     detected("face-recognized", True, name)
                    #     cv2.imwrite(
                    #         f'{self.authorize_output}/{datetime.now().strftime("%d_%m_%Y_%H_%M")}_{name}.jpg',
                    #         frame
                    #     )
                    # - End - 

                    cv2.putText(frame, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, color, 2)

            cv2.imshow("Face Recognition", frame)
                    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        is_ready("face-recognized", False)
        await detected("face-recognized", False, name, None)
        #detected("face-recognized", False, name)
        cap.release()
        cv2.destroyAllWindows()