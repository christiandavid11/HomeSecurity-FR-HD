import face_recognition
import pickle
import cv2
import os

from imutils import paths
    

class EncodeFaces:

    dataset = 'dataset'
    encodings = 'encodings.pickle'

    def __init__(self, detection_method='hog'):
        self.detection_method = detection_method


    def encode_faces(self):
        # for filename in os.listdir(self.dataset):
        #     file_path = os.path.join(self.dataset, filename)
        #     try:
        #         if os.path.isdir(file_path):
        #             os.rmdir(file_path)
        #             print("Folder deleted")
        #     except Exception as e:
        #         print(f"Failed: {e}")

        try:
            print("[INFO] Quantifying faces...")
            imagePaths = list(paths.list_images(self.dataset))

            knownEncodings = []
            knownNames = []

            for (i, imagePath) in enumerate(imagePaths):
                print("[INFO] Processing image {}/{}".format(i + 1, len(imagePaths)))
                name = imagePath.split(os.path.sep)[-2]

                image = cv2.imread(imagePath)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                boxes = face_recognition.face_locations(rgb, model=self.detection_method)

                encodings = face_recognition.face_encodings(rgb, boxes)

                for encoding in encodings:
                    knownEncodings.append(encoding)
                    knownNames.append(name)
        except:
            print("[ERROR] Something happened...")
        else:
            print("[INFO] Serializing encodings...")
            data = {"encodings": knownEncodings, "names": knownNames}
            f = open(self.encodings, "wb")
            f.write(pickle.dumps(data))
            f.close()