from datetime import datetime

import cv2
import numpy as np

from config import settings
from modules import detected, get_classes, is_ready


class HumanDetection:
    
    net = cv2.dnn.readNet(settings.yolo_weights, settings.yolo_cfg)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(320, 320), scale=1/255)

    def __init__(self, video_channel=0, roi=None, output_name=None):
        self.output_name = f'output/video/{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}_output.avi' \
            if output_name is None else output_name
        self.video_channel = video_channel
        self.roi = roi
        self.classes = get_classes()


    async def check_intersection(self, a, b):
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0] + a[2], b[0] + b[2]) - x
        h = min(a[1] + a[3], b[1] + b[3]) - y

        return False if w < 0 or h < 0 else True


    async def detection(self):
        await is_ready("human-detected", True)

        cap = cv2.VideoCapture(self.video_channel)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)        
        out = cv2.VideoWriter(self.output_name, cv2.VideoWriter_fourcc(*"XVID"), 20.0, (1280,720))
        res = []

        while True:
            ret, frame = cap.read()
            color = (255, 255, 255)
            
            if not (cap.isOpened and ret):
                break

            if self.roi is None:
                self.roi = cv2.selectROI('roi', frame)
                cv2.destroyWindow('roi')
            
            (class_ids, scores, bboxes) =  self.model.detect(frame)

            for class_id, score, bbox in zip(class_ids, scores, bboxes):
                class_name = self.classes[class_id]
                
                if class_name == "person":
                    res = [await self.check_intersection(np.array(box), np.array(self.roi)) for box in bboxes]
     
                cv2.rectangle(frame,(self.roi[0],self.roi[1]), (self.roi[0]+self.roi[2],self.roi[1]+self.roi[3]), color, 2)

            if any(res):
                await detected("human-detected", True)
            else:
                await detected("human-detected", False)

            out.write(frame)
            cv2.imshow("Human Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        await detected("human-detected", False)
        await is_ready("human-detected", False)
        out.release()
        cap.release()
        cv2.destroyAllWindows()