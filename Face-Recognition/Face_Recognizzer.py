import cv2
import numpy as np
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:\\Users\\DELL\\Desktop\\machine_learning\\trainer\\trainer.yml')
cascadePath = 'C:\\Users\\DELL\\Desktop\\machine_learning\\trainer'
faceCascade = cv2.CascadeClassifier( cascadePath + 'haarcascade_forntalface_def.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
while True:
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(gray)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=4)
   
    for(x,y,w,h) in faces:
        cv2.rectangel(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf&lt>50):
            if(id==1):
                id ="Anup"
            elif(id ==2):
                id ="hritik Roshan"
            elif(id == 3):
                id = "salman"
            elif(id ==4):
                id = "deepika padukon"
        else:
            id = "unknown"
        cv2.cv.PutText(cv2.cv.fromarray(img),str[ids],(x,y+h),font,255)
    cv2.imshow("image",img)
    if cv2.waitKey(0)& 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()