import numpy as np
from datetime import datetime
import cv2

import face_recognition
import pickle
import cv2
from datetime import datetime
from krakenio import Client


from modules import YOLO_CFG, YOLO_WEIGHTS, detected, get_classes, is_ready

class HumanDetection:
    
    net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(320, 320), scale=1/255)
########################### START ###########################
    encodings='encodings.pickle'
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    data = pickle.loads(open(encodings, "rb").read())
    recognize = 'output/authorize'
    unrecognize = 'output/unauthorize'
########################## END ###############################
    def __init__(self, video_channel=2, roi=None, output_name=None, fvideo_channgel = 0, foutput='output/video.avi', detection_method='hog'):
        self.output_name = f'output/video/{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}_output.avi' \
            if output_name is None else output_name
        self.video_channel = video_channel
        self.classes = get_classes()
        self.roi = roi
############################# START #############################
        self.foutput = foutput
        self.fvideo_channel = fvideo_channgel
        self.detection_method = detection_method
        self.authorize_output = 'output/authorize'
        self.unauthorize_output = 'output/unauthorize'

        is_ready("face-recognized", True)
################################ END #############################
        is_ready("human-detected", True)

    async def check_intersection(self, a, b):
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0] + a[2], b[0] + b[2]) - x
        h = min(a[1] + a[3], b[1] + b[3]) - y

        return False if w < 0 or h < 0 else True

    async def detection(self):
        cap = cv2.VideoCapture(self.video_channel)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)        
        out = cv2.VideoWriter(self.output_name, cv2.VideoWriter_fourcc(*"XVID"), 20.0, (1280,720))
###################### START #############################
        fcap = cv2.VideoCapture(self.fvideo_channel)
####################### END ##############################
        #global res


        while True:
            ret, frame = cap.read()
            #color = (255, 255, 255)
            res=[]
###################
            fret, fframe = fcap.read()
            color = (0, 255, 0)
###################       
            if not (cap.isOpened and ret):
                break

            if self.roi is None:
                
                self.roi = cv2.selectROI('roi', frame)
                cv2.destroyWindow('roi')

            if fret is False:
                print('[ERROR] Something wrong with your camera...')
                break
         
            boxes = face_recognition.face_locations(rgb, model=self.detection_method)
            encodings = face_recognition.face_encodings(rgb, boxes)
            
            names = []
            (class_ids, scores, bboxes) =  self.model.detect(frame)
            
            # for class_id, score, bbox in zip(class_ids, scores, bboxes):
            #     class_name = self.classes[class_id]   
            #     if class_name == "person":
            #         res = [await self.check_intersection(np.array(box), np.array(self.roi)) for box in bboxes]

            #if boxes and encodings:
                

            if any(res):
                await detected("human-detected", True, None)
                print("Alert sent")
            else:
                await detected("human-detected", False, None)

            

            out.write(frame)
            cv2.imshow("Human Detection", frame)
            ############
            cv2.imshow("Face Recognition", fframe)
            ############

            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        is_ready("human-detected", False)
        await detected("human-detected", False)
        ##########################
        is_ready("face-recognized", False)
        #await detected("face-recognized", False, name, None)
        ##########################
        out.release()
        cap.release()
        fcap.release()
        cv2.destroyAllWindows()