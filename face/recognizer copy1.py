import time
import face_recognition
import pickle
import cv2
from datetime import datetime
import numpy as np

#import base64
from krakenio import Client
import asyncio




from modules import detected, is_ready,YOLO_CFG, YOLO_WEIGHTS,get_classes

class FaceRecognition:
    net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(320, 320), scale=1/255)

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
        self.classes = get_classes()
        is_ready("face-recognized", True)
        is_ready("human-detected", True)


    async def face_recognize(self):
        cap = cv2.VideoCapture(self.video_channel)
        
        output_name = None
        output_name = f'output/video/{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}_output.avi' \
            if output_name is None else output_name

        cap_human = cv2.VideoCapture(2)
        cap_human.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap_human.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  
        out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*"XVID"), 20.0, (1280,720))

        writer = None

        from human.detection import HumanDetection
        roi = None


        while True:
            hd = HumanDetection()
            # await hd.detection()
            ret, frame = cap.read()
            ret1, frame1 = cap_human.read()
            color = (0, 255, 0)

            if ret is False:
                print('[ERROR] Something wrong with your camera...')
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r = frame.shape[1] / float(rgb.shape[1])

            boxes = face_recognition.face_locations(rgb, model=self.detection_method)
            encodings = face_recognition.face_encodings(rgb, boxes)
            
            names = []
            res = []

            if not (cap_human.isOpened and ret1):
                break

            if roi is None:
                roi = cv2.selectROI('roi', frame1)
                cv2.destroyWindow('roi')
            
            (class_ids, scores, bboxes) =  self.model.detect(frame)
            #(class_ids, scores, bboxes) =  self.model.detect(frame)

            for class_id, score, bbox in zip(class_ids, scores, bboxes):
                class_name = self.classes[class_id]
                
                if class_name == "person":
                    res = [hd.check_intersection(np.array(box), np.array(roi)) for box in bboxes]
                
                #cv2.rectangle(frame,(self.roi[0],self.roi[1]), (self.roi[0]+self.roi[2],self.roi[1]+self.roi[3]), color, 2)
            # from face.recognizer import FaceRecognition


            # if any(res):
            #     detected("human-detected", True)
            # else:
            #     detected("human-detected", False)

            if any(res):
                string_photo = None
                await detected("human-detected", True, string_photo)
            else:
                string_photo = None
                await detected("human-detected", False, string_photo)


            out.write(frame1)
            cv2.imshow("Human Detection", frame1)

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
                        # detected("face-recognized", False, name)
                        # color = (0, 0, 255)
                        # cv2.imwrite(
                        #     f'{self.unauthorize_output}/{datetime.now().strftime("%d_%m_%Y_%H_%M")}_{name}.jpg',
                        #     frame
                        # )
                        color = (0, 0, 255)
                        photo = f'{self.authorize_output}/{datetime.now().strftime("%d_%m_%Y_%H_%M")}_{name}.jpg'
                        cv2.imwrite(
                            photo,
                            frame
                        )
                        #string_photo = ""
                        # with open(photo, "rb") as img_file:
                        #     string_photo = base64.b64encode(img_file.read())
                        await detected("face-recognized", False, name, string_photo)

                        ###
                        # hd = HumanDetection()
                        # await hd.detection()
                        ###
                    else:
                        # detected("face-recognized", True, name)
                        # picx = cv2.imwrite(
                        #     f'{self.authorize_output}/{datetime.now().strftime("%d_%m_%Y_%H_%M")}_{name}.jpg',
                        #     frame
                        # )
                        # with open(picx, "rb") as img_file:
                        #     my_string = base64.b64encode(img_file.read())
                        # print(my_string)
                        
                        photo = f'{self.authorize_output}/{datetime.now().strftime("%d_%m_%Y_%H_%M")}_{name}.jpg'
                        cv2.imwrite(
                            photo,
                            frame
                        )

                        string_photo = ""
                        # with open(photo, "rb") as img_file:
                        #     string_photo = base64.b64encode(img_file.read())
                        await detected("face-recognized", True, name, string_photo)


                        ###
                        # hd = HumanDetection()
                        # await hd.detection()
                        ###

                    cv2.putText(frame, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, color, 2)


            cv2.imshow("Face Recognition", frame)
                    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # is_ready("face-recognized", False)
        # detected("face-recognized", False, name)
        # is_ready("human-detected", False)
        # detected("human-detected", False)
        
        is_ready("face-recognized", False)
        await detected("face-recognized", False, name, None)
        is_ready("human-detected", False)
        await detected("human-detected", False, name, None)

        cap.release()
        cv2.destroyAllWindows()




# import time
# import face_recognition
# import pickle
# import cv2
# from datetime import datetime

# from modules import detected, is_ready

# class FaceRecognition:

#     encodings='encodings.pickle'
#     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#     data = pickle.loads(open(encodings, "rb").read())
#     recognize = 'output/authorize'
#     unrecognize = 'output/unauthorize'

#     #def __init__(self, video_channel=0, output='output/video.avi', detection_method='hog'):
#     def __init__(self, video_channel=1, output='output/video.avi', detection_method='hog'):   
#         self.output = output
#         self.video_channel = video_channel
#         self.detection_method = detection_method
#         self.authorize_output = 'output/authorize'
#         self.unauthorize_output = 'output/unauthorize'

#         is_ready("face-recognized", True)


#     def face_recognize(self):
#         cap = cv2.VideoCapture(self.video_channel)
#         writer = None

#         while True:
#             ret, frame = cap.read()
#             color = (0, 255, 0)

#             if ret is False:
#                 print('[ERROR] Something wrong with your camera...')
#                 break

#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             r = frame.shape[1] / float(rgb.shape[1])

#             boxes = face_recognition.face_locations(rgb, model=self.detection_method)
#             encodings = face_recognition.face_encodings(rgb, boxes)
            
#             names = []
            
#             if boxes and encodings:
#                 for encoding in encodings:
#                     matches = face_recognition.compare_faces(self.data["encodings"], encoding)
#                     name = "Unknown"

#                     if True in matches:
#                         matchedIdxs = [i for (i, b) in enumerate(matches) if b]
#                         counts = {}

#                         for i in matchedIdxs:
#                             name = self.data["names"][i]
#                             counts[name] = counts.get(name, 0) + 1

#                         name = max(counts, key=counts.get)

#                     names.append(name)

#                 for ((top, right, bottom, left), name) in zip(boxes, names):
#                     top = int(top * r)
#                     right = int(right * r)
#                     bottom = int(bottom * r)
#                     left = int(left * r)

#                     if name == "Unknown":
#                         detected("face-recognized", False, name)
#                         color = (0, 0, 255)
#                         cv2.imwrite(
#                             f'{self.unauthorize_output}/{datetime.now().strftime("%d_%m_%Y_%H_%M")}_{name}.jpg',
#                             frame
#                         )
#                     else:
#                         detected("face-recognized", True, name)
#                         cv2.imwrite(
#                             f'{self.authorize_output}/{datetime.now().strftime("%d_%m_%Y_%H_%M")}_{name}.jpg',
#                             frame
#                         )

#                     cv2.putText(frame, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.75, color, 2)

#             cv2.imshow("Face Recognition", frame)
                    
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         is_ready("face-recognized", False)
#         detected("face-recognized", False, name)
#         cap.release()
#         cv2.destroyAllWindows()