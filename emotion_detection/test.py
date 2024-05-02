import cv2
import numpy as np
import os
from keras.models import load_model

face_classifier=cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

path='project/emotion detection/Final_Resnet50_Best_model.keras'
isFile = os.path.isfile(path) 
classifier=load_model(isFile)

emotion_labels=['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap=cv2.VideoCapture(0)

while True:
     
     _, frame= cap.read()

     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

     faces= face_classifier.detectMultiScale(gray)

     for(x,y,w,h) in faces:
          cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

          roi_gray = gray[y:y+h, x:x+w]

          roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

          if np.sum([roi_gray]) !=0:
               roi =roi_gray.astype('float')/255.0
               roi.expand_dims(roi,axis=0)

               predicition = classifier.predict(roi)[0]
               label=emotion_labels[predicition.argmax()]
               label_position = (x,y)

          else:
               
               cv2.putText(frame,'No Faces',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        
     cv2.imshow('Emotion Detector',frame)

     if cv2.waitKey(1) & 0xFF == ord('q'):
          break


cap.release()
cv2.destroyAllWindows()


