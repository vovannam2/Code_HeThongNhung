import cv2
import numpy as np
stop_sign = cv2.CascadeClassifier("stopsign_classifier.xml")
turn_right = cv2.CascadeClassifier("turnRight_ahead.xml")
turn_left = cv2.CascadeClassifier("turnLeft_ahead.xml")

class Detection(object):
        def signDetected(self,image):
                h,w = image.shape[:2]
                gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                Stop = stop_sign.detectMultiScale(gray,1.02,10)
                Turn_Right = turn_right.detectMultiScale(gray,1.02,10)
                Turn_Left = turn_left.detectMultiScale(gray,1.02,10)
                
                if len(Stop) > 0:
                        for (x,y,w,h) in Stop:
			        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
      
                if len(Turn_Right) > 0:
                        for (x,y,w,h) in Turn_Right:
				cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                                
                if len(Turn_Left) > 0:
                        for (x,y,w,h) in Turn_Left:
				cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

                
