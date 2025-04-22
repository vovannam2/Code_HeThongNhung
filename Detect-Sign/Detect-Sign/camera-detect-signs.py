import cv2
import time
import numpy as np
import tensorflow as tf
 
#############################################
frameHeight = 240
frameWidth= 480         # CAMERA RESOLUTION
brightness = 180
threshold = 0.75        # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################
 
# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# # IMPORT THE TRANNIED MODEL
# pickle_in = open("model_trained.p","rb")
# model = pickle.load(pickle_in)

# IMPORT THE TRAINED MODEL USING CHECKPOINT
checkpoint_path = "model-checkpoint"
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4)  # Assuming 4 classes (Stop, Right, Left, Straight)
])

# Initialize the checkpoint
checkpoint = tf.train.Checkpoint(model=model)

# Restore the latest checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
print("Model restored from checkpoint")
 
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
    # READ IMAGE
    success, imgOrignal = cap.read()
    
    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)

    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.amax(predictions)

    print(predictions)
    print(getCalssName(classIndex))
    print(probabilityValue)
    
    if probabilityValue > threshold:
        cv2.putText(imgOrignal, str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Result", imgOrignal)
    if cv2.waitKey(1) == ord('q'):
        break