import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

imgElon = face_recognition.load_image_file('images/obama.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('images/obama2.jpg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgElon)[0]
encodeobama = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloctest = face_recognition.face_locations(imgtest)[0]
encodeobamatest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeobama],encodeobamatest)
face_dis = face_recognition.face_distance([encodeobama],encodeobamatest)
print(results,face_dis)
cv2.putText(imgtest,f'{results} {round(face_dis[0],2)}' ,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Obama',imgElon)
cv2.imshow('Obama2',imgtest)
cv2.waitKey(0)