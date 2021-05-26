import cv2
import numpy as np
from PIL import Image 
import os
path =('C:\\Users\\DELL\\Desktop\\machine_learning\\dataSet')
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('C:\\Users\\DELL\\Desktop\\machine_learning\\haarcascade_frontalface_default.xml');
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []
    
    for imagePath in imagePaths:
        
        PIL_img = Image.open(imagePath).convert('L')
#convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
    
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        
        for (x,y,w,h) in faces:
            
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print("Training Faces.....")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
recognizer.save('trainer/trainer.yml')
print(recognizer)
print("\n faces trained ".format(len(np.unique(ids))))