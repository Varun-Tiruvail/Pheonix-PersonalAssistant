import cv2
import numpy as np
from PIL import Image as img
import os

path = 'UserFaceData'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def Images_And_Labels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:
        gray_img = img.open(imagePath).convert('L')
        img_arr = np.array(gray_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_arr)
        for (x,y,w,h) in faces:
            faceSamples.append(img_arr[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids

print ("Training face data. Please Wait ...")
faces,ids = Images_And_Labels(path)
recognizer.train(faces, np.array(ids))
recognizer.write('ModelTrainer/trainer.yml')
print("User Face is succesfully stored to recognize.")