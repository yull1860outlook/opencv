import cv2
import threading
from deepface import DeepFace
from deepface.detectors import OpenCvWrapper

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

counter = 0
face_match=False
reference_img = cv2.imread("refme.jpg")

#FACE VERIFICATION MODELS
models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
]

#FACE DETECTORS MODELS
backends = [
  'opencv',   #haarcascade
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
]


#  based on 
#   OpenCvWrapper.align_face(frame)
#
def detect_eye(frame):   
    
    detected_face_gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )  # eye detector expects gray scale image

    cv2.imwrite("gray.jpg",frame)
    detctor = OpenCvWrapper.build_model()
    # eyes = eye_detector.detectMultiScale(detected_face_gray, 1.3, 5)
    eyes = detctor['eye_detector'].detectMultiScale(detected_face_gray, 1.1, 10)

    # ----------------------------------------------------------------

    # opencv eye detectin module is not strong. it might find more than 2 eyes!
    # besides, it returns eyes with different order in each call (issue 435)
    # this is an important issue because opencv is the default detector and ssd also uses this
    # find the largest 2 eye. Thanks to @thelostpeace

    eyes = sorted(eyes, key=lambda v: abs(v[2] * v[3]), reverse=True)

    # ----------------------------------------------------------------

    if len(eyes) >= 2:
        # decide left and right eye

        eye_1 = eyes[0]
        eye_2 = eyes[1]

        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1

        # -----------------------
        # find center of eyes
        left_eye = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        right_eye = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
        #img = FaceDetector.alignment_procedure(img, left_eye, right_eye)
        eyes2return = [left_eye, right_eye]
        return eyes2return
    else:
        return None


def check_face(frame):
    global face_match
    try:
        #if DeepFace.verify(img1_path=frame,img2_path=reference_img)["verified"]:
        
        #face recognition based on DB (images)        
        #dfs = DeepFace.find(img_path = frame, 
        #    db_path = "./my_db", 
        #    detector_backend = backends[1]
        #    )
        #org = frame.copy()
        dfs = DeepFace.extract_faces(frame,detector_backend= backends[0],enforce_detection=False,grayscale=False,align=False)
        if len(dfs) > 0:
        #if 0:
            #print(dfs)
            item=1
            for df in dfs:
                if df['confidence'] < 0.5:
                    continue

                faces =[]
                faces.append((df['facial_area']['x'],df['facial_area']['y'],df['facial_area']['w'],df['facial_area']['h']))
                #print(rec)
                for x,y,w,h in faces:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                    imgcheckeye = frame[y : y+h, x : x+w]              
                    eyes = detect_eye(imgcheckeye)
                    if eyes is not None:
                        for (ex,ey) in eyes:
                            cv2.circle(frame,(ex+x,ey+y),2,(0,255,0),2)
                            
                cv2.putText(frame,"score:"+str(df['confidence']),(20,20*item),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                item+=1
                
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False


while True:
    ret , frame = cap.read()
    if ret : 
        frame=cv2.flip(frame,0)
        if counter %2 == 0:
            try:
                #threading.Thread(target = check_face,args=(frame.copy(),)).start()
                check_face(frame)
                cv2.imshow("Video",frame)
            except ValueError:
                pass

        counter += 1

        #if face_match:
            # startpt , font , font size, color (BGR), thickness
        #    cv2.putText(frame," ** MATCH ** ",(20,400),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        #else:
        #    cv2.putText(frame," **No MATCH** ",(20,400),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)


    key = cv2.waitKey(1)
    if key == ord("q"):
        cv2.imwrite("last.jpg",frame)
        break


cv2.destroyAllWindows()