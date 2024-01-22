import sys
import asyncio

if __name__ == '__main__':
    args = sys.argv[1]

    if args == "train":
        from face.encoder import EncodeFaces
        from modules import get_images

        get_images()
        ef = EncodeFaces()
        ef.encode_faces()
    
    elif args == "run":
        from face.recognizer import FaceRecognition
        
        fr = FaceRecognition()
        # fr.face_recognize()
        asyncio.run(fr.face_recognize())

    else:
        import cv2
        import time

        cap = cv2.VideoCapture(0)
        pTime = 0
        while True:
            ret, frame = cap.read()

            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime=cTime

            if ret is False:
                break

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 2), 2)

            cv2.imshow('frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    