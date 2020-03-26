#!/usr/bin/env python
import cv2 
import RPi.GPIO as GPIO
import numpy as np
from PIL import Image
import os
import argparse
import MySQLdb
import time

#Thiết lập động cơ Servo
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7,GPIO.OUT)
#Disable warnings (optional)
GPIO.setwarnings(False)
#chân số 7 (GPIO4)
servo=GPIO.PWM(7,50)
#động cơ lúc bắt đầu sẽ ở vị trí khóa cửa.
servo.start(2.5)

#Thiet lap coi buzzer
pin = 11
GPIO.setup(pin,GPIO.OUT)
buzzer = GPIO.PWM(pin, 100)
buzzer.start(99)
#Ham dem so lan nhan dien sai
#count=0

#Thiết lập đè Led và nút bấm
GPIO.setmode(GPIO.BOARD)

# Đèn cho ta biết là camera đang quét mặt
ledPin = 12

# Chân đèn xanh báo của đã mở
ledPinOpen = 31

# Chân đèn đỏ báo không nhận diện được khuôn mặt
ledPinWarning = 33

# Chân nút nhấn để cho camera quét mặt
buttonPin = 16

# Chân nút đóng cửa
buttonPinClose = 18

# Chân nút mở cửa
buttonPinOpen = 22

GPIO.setup(ledPin, GPIO.OUT)
GPIO.setup(ledPinOpen, GPIO.OUT)
GPIO.setup(ledPinWarning, GPIO.OUT)
GPIO.setup(buttonPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(buttonPinClose, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(buttonPinOpen, GPIO.IN, pull_up_down=GPIO.PUD_UP)

#Thiết lập lưu dữ liệu vào Cơ sở dữ liệu Mysql
db = MySQLdb.connect("localhost", "monitor", "password", "USER")
curs=db.cursor()

#Load pretrained cascade face detection -- Load file cơ sở nhận diện khuôn mặt (OpenCV)
face_cascade = cv2.CascadeClassifier("/home/pi/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml")

#Load improved FPS video instead of the default Picamera -- tải video FPS được cải thiện thay vì Picamera mặc định
cam = cv2.VideoCapture(0)
#Load the recognizer -- Tải file nhận dạng
rec = cv2.face.createLBPHFaceRecognizer()
#Load trained local binary pattern face data -- Tải dữ liệu  dữ liệu khuôn mặt dạng nhị phân đã được train
rec.load("recognizer/trainingData.yml")
id=0
font = cv2.FONT_HERSHEY_SIMPLEX

#Face recognition function -- Hàm nhận diện khuôn mặt
def detectFace(faces,hog,img):
    for (x, y, w, h) in faces:
        #Tạo khuôn hình vuông để hiện thị
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        result = cv2.face.MinDistancePredictCollector()
        rec.predict(hog[y:y+h,x:x+w],result, 0)
        id = result.getLabel()
        
        conf = result.getDist()
        if(conf<50):
            if(id==1):
                id="Tu_"+str(conf)
                #Lưu lại thông tin người dùng vào Cơ sở dữ liệu nếu nhận diện thành công.
                sql = """INSERT INTO Detail(ID, Name, tdate, ttime) values('1','Tu', CURRENT_DATE(), NOW())"""
                curs.execute(sql)
                #Lưu ảnh của người được nhận và thư mục DataAsset với định dạng User.id_người_sử_dụng.jpg .
                cv2.imwrite("DataAsset/User."+id +'.' + ".jpg", gray[y:y+h,x:x+w])
                #2s luu vao Database 1 lan.
                #time.sleep(2)
                #Commit lên Mysql
                db.commit()
                
                #Động cơ Servo sẽ mở cửa.
                servo.ChangeDutyCycle(5.5)
                GPIO.output(ledPinOpen, GPIO.HIGH)
                time.sleep(1)
                GPIO.output(ledPinOpen, GPIO.LOW)
                
                #servo.stop()
#                servo.ChangeDutyCycle(5.5)
#                time.sleep(2)
                
            elif(id==2):
                id="Thang_"+str(conf)
                #Lưu lại thông tin người dùng vào Cơ sở dữ liệu nếu nhận diện thành công
                sql = """INSERT INTO Detail(ID, Name, tdate, ttime) values('2','Thang', CURRENT_DATE(), NOW())"""
                curs.execute(sql)
                #Lưu ảnh của người được nhận.
                cv2.imwrite("DataAsset/User."+id+'.' + ".jpg", gray[y:y+h,x:x+w])
                #time.sleep(2)
                db.commit()
                
                servo.ChangeDutyCycle(5.5)
                GPIO.output(ledPinOpen, GPIO.HIGH)
                time.sleep(1)
                GPIO.output(ledPinOpen, GPIO.LOW)
                
                #servo.stop()
#                servo.ChangeDutyCycle(5.5)
#                time.sleep(2)
        else:
            id="Unknow_" + str(conf)
            sql = """INSERT INTO Detail(ID, Name, tdate, ttime) values('3','Unknow', CURRENT_DATE(), NOW())"""
            curs.execute(sql)
            #Lưu ảnh của người được nhận.
            cv2.imwrite("DataAsset/Unknow"+id+'.' + ".jpg", gray[y:y+h,x:x+w])
            #time.sleep(2)
            db.commit()
            GPIO.output(ledPinWarning, GPIO.HIGH)
            time.sleep(1)
            GPIO.output(ledPinWarning, GPIO.LOW)
            buzzer.ChangeFrequency(1000)
            time.sleep(1)
            buzzer.ChangeFrequency(1)
            #count = count + 1
            #if (count == 3) :
            #    buzzer.ChangeFrequency(1000)
            #    time.sleep(2)
            #    buzzer.ChangeFrequency(1)
            #    count = 0
            
        cv2.putText(img,str(id),(x,y+h),font,1,(255,255,255),2,cv2.LINE_AA)      
       
#Hàm lấy ảnh.
def get_image():
    return cam.read()[1]
while(True):
    #read each frame in the real-time video -- Đọc từng Khung hình trực tiếp.

    buttonState = GPIO.input(buttonPin)
    buttonStateClose = GPIO.input(buttonPinClose)
    buttonStateOpen = GPIO.input(buttonPinOpen)
    if buttonStateClose == False:
        servo.ChangeDutyCycle(2.5)
        
    if buttonStateOpen == False:
        servo.ChangeDutyCycle(5.5)
        
    frame = get_image();
    img = frame;
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(equ, 1.3, 5)
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
            #Nút được nhấn mới bắt đầu nhận diện và đèn led sáng.
            if buttonState == False:
                GPIO.output(ledPin, GPIO.HIGH)
                detectFace(faces,dst,img)
            else:
                GPIO.output(ledPin, GPIO.LOW)
        else:
            #Nút được nhấn mới bắt đầu nhận diện và đèn led sáng
            if buttonState == False:
                GPIO.output(ledPin, GPIO.HIGH)
                detectFace(faces,dst,img)
            else:
                GPIO.output(ledPin, GPIO.LOW)
    else:
        if buttonState == False:
            GPIO.output(ledPin, GPIO.HIGH)
            detectFace(faces,equ,img)
        else:
            GPIO.output(ledPin, GPIO.LOW)
    cv2.imshow('Face', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("cleanup")
GPIO.cleanup()
cam.release()
cv2.destroyAllWindows()


