import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

video_capture = cv2.VideoCapture(0)

while True:
	ret, frame = video_capture.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	equ = cv2.equalizeHist(gray) 
	faces = face_cascade.detectMultiScale(equ, 1.05, 5,minSize=(10,10))
	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)

	cv2.imshow('video',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
video_capture.release()
cv2.destroyAllWindows()
