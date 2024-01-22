import asyncio
import pickle
import time
from datetime import datetime

import cv2
import face_recognition
import numpy as np

from config import settings
from modules import detected, draw_border, is_ready


class FaceRecognition:

    encodings = settings.encodings
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    data = pickle.loads(open(encodings, "rb").read())


    def __init__(self, video_channel=0, output='output/video.avi', detection_method='hog'):
        self.output = output
        self.video_channel = video_channel
        self.detection_method = detection_method
        self.authorize_output = 'output/authorize'
        self.unauthorize_output = 'output/unauthorize'

    async def face_recognize(self):
        await is_ready("face-recognized", True)
        cap = cv2.VideoCapture(self.video_channel)

        boxes = []
        encodings = []
        names = []
        process_this_frame = True

        while True:
            ret, frame = cap.read()

            if ret is False:
                print('[ERROR] Something wrong with your camera...')
                break

            if process_this_frame:
                small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25) # type: ignore

                rgb_small_frame = small_frame[:,:,::-1]

                boxes = face_recognition.face_locations(rgb_small_frame)
                encodings = face_recognition.face_encodings(rgb_small_frame, boxes)

                names = []
                name = "Unknown"
                color = (0, 0, 255)
                for encoding in encodings:
                    matches = face_recognition.compare_faces(self.data["encodings"], encoding)
                    
                    face_distances = face_recognition.face_distance(self.data["encodings"], encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.data["names"][best_match_index]
                        color = (0, 255, 0)

                    names.append(name)

            process_this_frame = not process_this_frame

            for (top, right, bottom, left), name in zip(boxes, names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                await draw_border(frame, (left, top), (right, bottom), color, 2, 10, 20) # type: ignore
                cv2.putText(frame, name, (left, top - 5), cv2.FONT_HERSHEY_DUPLEX, .75, color, 1) # type: ignore
            
                if name == "Unknown":
                    await detected("face-recognized", False, name)
                    cv2.imwrite(f'{self.unauthorize_output}/{datetime.now().strftime("%d_%m_%Y_%H_%M")}_{name}.jpg', frame)
                else:
                    await detected("face-recognized", True, name)
                    cv2.imwrite(f'{self.authorize_output}/{datetime.now().strftime("%d_%m_%Y_%H_%M")}_{name}.jpg', frame)
            
            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        await is_ready("face-recognized", False)
        await detected("face-recognized", False, name)# type: ignore

        cap.release()
        cv2.destroyAllWindows()