import cv2
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2

#############################################
frameWidth = 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75         # PROBABILITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# LOAD TFLITE MODEL
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

def getClassName(classNo):
    if classNo == 0:
        return 'Stop'
    elif classNo == 1:
        return 'Right'
    elif classNo == 2:
        return 'Left'
    elif classNo == 3:
        return 'Straight'

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

width, height = 480, 240
    cap = Picamera2()
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
        cv2.putText(img, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        
        # PREDICT IMAGE
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        classIndex = np.argmax(predictions)
        probabilityValue =np.amax(predictions)

        if probabilityValue > threshold:
            cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Result", imgOrignal)
     
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break