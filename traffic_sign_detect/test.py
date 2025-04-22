import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from picamera2 import Picamera2
 
#############################################
frameWidth= 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################
 
model = load_model('my_trained_model.h5')
 
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getCalssName(classNo):
    if   classNo == 0: return 'Stop'
    elif classNo == 1: return 'Right'
    elif classNo == 2: return 'Left'
    elif classNo == 3: return 'Straight'
 
while True:
    width, height = 480, 240
    imgOrignal = Picamera2()
    cap.preview_configuration.main.size = (width, height)
    cap.preview_configuration.main.format = "RGB888"
    cap.preview_configuration.align()
    cap.configure("preview")
    cap.start()
    
    while True: 
        img = cap.capture_array()
        img = cv2.resize(img, (32, 32))
        
        img = preprocessing(img)
        cv2.imshow("Processed Image", img)

        img = img.reshape(1, 32, 32, 1)
        cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        
        # PREDICT IMAGE
        predictions = model.predict(img)
        classIndex = np.argmax(predictions)
        probabilityValue =np.amax(predictions)

        if probabilityValue > threshold:
            cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Result", imgOrignal)
     
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
