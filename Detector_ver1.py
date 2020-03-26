#!/usr/bin/env python
import cv2 
import RPi.GPIO as GPIO
import numpy as np
from PIL import Image
import os
import argparse
import MySQLdb
import time


db = MySQLdb.connect("localhost", "monitor", "password", "USER")
curs=db.cursor()

#Load pretrained cascade face detection
face_cascade = cv2.CascadeClassifier("/home/pi/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml")
#Load improved FPS video instead of the default Picamera
cam = cv2.VideoCapture(0)
#Load the recognizer
rec = cv2.face.createLBPHFaceRecognizer()
#Load trained local binary pattern face data
rec.load("recognizer/trainingData.yml")
id=0
font = cv2.FONT_HERSHEY_SIMPLEX

#Face recognition function
def detectFace(faces,hog,img):
    for (x, y, w, h) in faces:
       cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
       result = cv2.face.MinDistancePredictCollector()
       rec.predict(hog[y:y+h,x:x+w],result, 0)
       id = result.getLabel()

       conf = result.getDist()
       if(conf<50):
           if(id==1):
                id="Tu_"+str(conf)
                #Luu ten nguoi dung lai khi nhan dien thanh cong.
                sql = """INSERT INTO Detail(ID, Name, tdate, ttime) values('1','Tu', CURRENT_DATE(), NOW())"""
                curs.execute(sql)
                #2s luu vao Database 1 lan.
                #time.sleep(2)
                db.commit()
#                servo.ChangeDutyCycle(5.5)
#                time.sleep(2)
                
           elif(id==2):
                id="Thang_"+str(conf)
                sql = """INSERT INTO Detail(ID, Name, tdate, ttime) values('2','Thang', CURRENT_DATE(), NOW())"""
                curs.execute(sql)
                #time.sleep(2)
                db.commit()
#                servo.ChangeDutyCycle(5.5)
#                time.sleep(2)
       else:
            id="Unknow"
            sql = """INSERT INTO Detail(ID, Name, tdate, ttime) values('3','Unknow', CURRENT_DATE(), NOW())"""
            curs.execute(sql)
            #time.sleep(2)
            db.commit()
       cv2.putText(img,str(id),(x,y+h),font,1,(255,255,255),2,cv2.LINE_AA)      
       
def get_image():
    return cam.read()[1]
while(True):
    #read each frame in the real-time video
    frame = get_image();
    img = frame;
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(equ, 1.3, 5)
    #If the face is not frontal face then rotate the face by +30/-30 degree 
    if len(faces)==0:
        rows,cols = equ.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),30,1)
        dst = cv2.warpAffine(equ,M,(cols,rows))
        faces = face_cascade.detectMultiScale(dst, 1.3, 5)
        if len(faces)==0:
            rows,cols = equ.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2),-30,1)
            dst = cv2.warpAffine(equ,M,(cols,rows))
            faces = face_cascade.detectMultiScale(dst, 1.3, 5)
            detectFace(faces,dst,img)

        else:
            detectFace(faces,dst,img)

    else:
        detectFace(faces,equ,img)

    cv2.imshow('Face', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


