import numpy as np
from datetime import datetime
import cv2

from modules import YOLO_CFG, YOLO_WEIGHTS, detected, get_classes, is_ready

from time import sleep
import face_recognition
import pickle
from krakenio import Client

from gpiozero import Buzzer 
import easygui

import usb.core
import requests

l_url = 'https://towerofgod.onrender.com/camera/status'

class HumanDetection:
    
    net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(320, 320), scale=1/255)

    ### face ###
    encodings='encodings.pickle'
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    data = pickle.loads(open(encodings, "rb").read())
    recognize = 'output/authorize'
    unrecognize = 'output/unauthorize'

    ### end face ###

    def __init__(self, video_channel=0, roi=None, output_name=None, output='output/video.avi', detection_method='hog'):
        self.output_name = f'output/video/{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}_output.avi' \
            if output_name is None else output_name
        self.video_channel = video_channel
        self.classes = get_classes()
        self.roi = roi

        is_ready("human-detected", True)

        ### face ###
        self.output = output
        self.detection_method = detection_method
        self.authorize_output = 'output/authorize'
        self.unauthorize_output = 'output/unauthorize'
        self.classes = get_classes()
        is_ready("face-recognized", True)
        is_ready("human-detected", True)
        ### end face ###


    async def check_intersection(self, a, b):
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0] + a[2], b[0] + b[2]) - x
        h = min(a[1] + a[3], b[1] + b[3]) - y

        return False if w < 0 or h < 0 else True

    # def checkk(res):
    #     if any(res):
    #         detected("human-detected", True)


    async def detection(self):
        cap = cv2.VideoCapture(self.video_channel)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)        
        out = cv2.VideoWriter(self.output_name, cv2.VideoWriter_fourcc(*"XVID"), 20.0, (1280,720))

        vendor_id = 0x0c45
        product_id = 0x6367
        sStatus = 0
        #global res
        

        while True:
            ret, frame = cap.read()
            color = (255, 255, 255)
            res=[]
            
            if not (cap.isOpened and ret):
                break

            if self.roi is None:
                
                self.roi = cv2.selectROI('roi', frame)
                cv2.destroyWindow('roi')
            
            #(class_ids, scores, bboxes) =  self.model.detect(frame)
            (class_ids, scores, bboxes) =  self.model.detect(frame)

            ### check cam
            if (detect_usb_device(vendor_id, product_id) == True):
                detect_usb_device(vendor_id, product_id)
                # try:
                #     print("Hi")
                # except:
                #     print("hello")
            ### end check cam
            
            for class_id, score, bbox in zip(class_ids, scores, bboxes):
                class_name = self.classes[class_id]
                
                if class_name == "person":
                    res = [self.check_intersection(np.array(box), np.array(self.roi)) for box in bboxes]
                          
                #cv2.rectangle(frame,(self.roi[0],self.roi[1]), (self.roi[0]+self.roi[2],self.roi[1]+self.roi[3]), color, 2)

            if any(res):
                await detected("human-detected", True, None)
                print("Human Detected. Alert sent")
                
                ## facial recognition starts
                ## writer = None
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
                            buz_on()
                            #break
                            #return 

                        else:
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
                            print("Sending image")
                            await detected(type="face-recognized", is_detected=True, name=name, uploaded_file=img_url)
                            
                            print("Image sent")


                            cv2.putText(frame, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, color, 2)
                            break

            else:
                await detected("human-detected", False, None)

            out.write(frame)
            cv2.imshow("Human Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        await is_ready("human-detected", False)
        await detected("human-detected", False)
        is_ready("face-recognized", False)
        await detected("face-recognized", False, None, None)

        out.release()
        cap.release()
        cv2.destroyAllWindows()

########################################################################################
def buz_on():
    buzzer = Buzzer(21)
    file_path = "/home/pi/Desktop/human-detection-main/passcode.txt"
    file = open(file_path, "r")

    # Read the entire contents of the file
    file_contents = file.read()

    # Close the file
    file.close()

    while True:
        buzzer.on()
        pin = easygui.enterbox("Enter a 4-digit PIN:", title="PIN Entry")
        validate_pin(pin)
        if validate_pin(pin) == True and pin == file_contents:
            buzzer.off()
            break
        else:
            easygui.msgbox("INVALID PIN", title = "Please enter PIN")
            

def validate_pin(pin):
    if len(pin) == 4 and pin.isdigit():
        return True
    return False

def detect_usb_device(vendor_id, product_id):
    while True:
        # Find the USB device based on vendor and product IDs
        device = usb.core.find(idVendor=vendor_id, idProduct=product_id)

        if device is not None:
            print("USB Device connected!")
            return False
        else:
            print("USB Device disconnected!")
            myobj = {
                'status' :  1
            }
            q = requests.post(l_url, myobj)
            return True

        # Wait for a short duration before checking again
        # time.sleep(1)