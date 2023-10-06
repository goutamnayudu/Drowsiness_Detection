import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import pygame
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


pygame.mixer.init()
alert_sound = pygame.mixer.Sound('C:\\Users\\Visitor\\Desktop\\Drowsiness\\alarm.wav')  # Replace 'alarm.wav' with your own alert sound file

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:\\Users\\Visitor\\Desktop\\Drowsiness\\shape_predictor_68_face_landmarks.dat')  # Downloaded from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2


EYE_AR_THRESHOLD = 0.25  
EYE_AR_CONSEC_FRAMES = 30 
COUNTER = 0  
ALARM_ON = False  


def eye_aspect_ratio(eye):
    
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    
    C = dist.euclidean(eye[0], eye[3])
  
    ear = (A + B) / (2.0 * C)
    return ear


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    if(len(faces)>0):
        face=faces[0]
        shape = shape_to_np(predictor(gray, face))

        
        left_eye = shape[42:48]
        right_eye = shape[36:42]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 0, 255), 2)
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 0, 255), 2)

        if ear < EYE_AR_THRESHOLD:
            COUNTER += 1
            if ALARM_ON:
                if not (pygame.mixer.get_busy()):
                    alert_sound.play()
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                else:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            ALARM_ON = False
            pygame.mixer.Sound.stop(alert_sound)

        cv2.putText(frame, f"EAR: {ear:.2f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        

    cv2.imshow("Driver Drowsiness Detection", frame)
    


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
