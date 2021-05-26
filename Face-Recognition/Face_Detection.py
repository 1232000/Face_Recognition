import numpy as np
import cv2
faceCascade = cv2.CascadeClassifier('C:\\Users\\DELL\\Desktop\\machine_learning\\trainer\\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3,1260)
cap.set(3,720)
while True:
    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.2,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        gray = gray[y:y+h,x:x+w]
        print(gray)
        color = img[y:y+h,x:x+w]
    cv2.imshow('Video',img)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()