import cv2
import threading
from deepface import DeepFace

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
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
]

def check_face(frame):
    global face_match
    try:
        #if DeepFace.verify(img1_path=frame,img2_path=reference_img)["verified"]:
        
        #face recognition based on DB (images)        
        #dfs = DeepFace.find(img_path = frame, 
        #    db_path = "./my_db", 
        #    detector_backend = backends[1]
        #    )
        dfs = DeepFace.extract_faces(frame,target_size=(640,480),detector_backend= backends[4],enforce_detection=False,grayscale=False)
        if len(dfs) > 0:
        #if 0:
            #print(dfs)
            item=1
            for df in dfs:
                cv2.putText(frame,"score:"+str(df['confidence']),(20,20*item),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                rec=list(df['facial_area'].values())
                #print(rec)
                #for x,y,w,h in df['facial_area']:
                cv2.rectangle(frame,(rec[0],rec[1]),(rec[0]+rec[2],rec[1]+rec[3]),(0,0,255),3)
                item+=1
                
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False


while True:
    ret , frame = cap.read()
    if ret : 
        #frame=cv2.flip(frame,0)
        if counter %2 == 0:
            try:
                #threading.Thread(target = check_face,args=(frame.copy(),)).start()
                check_face(frame)
                cv2.imshow("Video",frame)
            except ValueError:
                pass

        counter += 1

        if face_match:
            # startpt , font , font size, color (BGR), thickness
            cv2.putText(frame," ** MATCH ** ",(20,400),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame," **No MATCH** ",(20,400),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)


    key = cv2.waitKey(1)
    if key == ord("q"):
        cv2.imwrite("last.jpg",frame)
        break


cv2.destroyAllWindows()