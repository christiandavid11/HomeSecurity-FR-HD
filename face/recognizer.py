import time
import face_recognition
import pickle
import cv2
from datetime import datetime
import numpy as np
from krakenio import Client


from modules import detected, is_ready, YOLO_CFG, YOLO_WEIGHTS, get_classes

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

    # def check_intersection(self, a, b):
    #     x = max(a[0], b[0])
    #     y = max(a[1], b[1])
    #     w = min(a[0] + a[2], b[0] + b[2]) - x
    #     h = min(a[1] + a[3], b[1] + b[3]) - y

    #     return False if w < 0 or h < 0 else True

    async def face_recognize(self):
        from human.detection import HumanDetection
        
        cap = cv2.VideoCapture(self.video_channel)
        
        output_name = None
        output_name = f'output/video/{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}_output.avi' \
            if output_name is None else output_name

        cap_human = cv2.VideoCapture(2)
        cap_human.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap_human.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  
        out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*"XVID"), 20.0, (1280,720))
        hd = HumanDetection()

        roi = None
        
        while True:
            start = time.time()
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

            for class_id, score, bbox in zip(class_ids, scores, bboxes):
                class_name = self.classes[class_id]
                
                if class_name == "person":
                    res = [hd.check_intersection(np.array(box), np.array(roi)) for box in bboxes]
                
            if any(res):
                string_photo = None
                await detected("human-detected", True, string_photo)
                print("Person detected")

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

                    cv2.putText(frame, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, color, 2)

            print("--- %s seconds ---" % (time.time() - start))
            cv2.imshow("Face Recognition", frame)
                    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        is_ready("face-recognized", False)
        await detected("face-recognized", False, None, None)
        is_ready("human-detected", False)
        await detected("human-detected", False, None, None)
        
        cap.release()
        cv2.destroyAllWindows()