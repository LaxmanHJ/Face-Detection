import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml')

img = cv2.imread('/path to image/person.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert image to grayscale
faces = face_cascade.detectMultiScale(gray, 1.1, 5)
#print(faces)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    #print(x,y,w,h)
    #print(eyes)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
cv2.imshow('original imgage',img)
cv2.imshow('orignal gray image',gray)
cv2.imshow('only colored face',roi_color)
cv2.imshow('only face in gray',roi_gray)

cv2.waitKey(0)
cv2.destroyAllWindows()





