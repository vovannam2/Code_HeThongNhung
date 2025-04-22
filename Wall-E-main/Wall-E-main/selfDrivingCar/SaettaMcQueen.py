import utils
from MotorModule import Motor
import RoadModule as Rm
import cv2, numpy as np
from picamera2 import Picamera2

##################################################
motor = Motor(25,23,24,16,21,20)
width, height = 480, 240
##################################################

def main():
    trip()
    # turnAround()
    # trip()

    # motor.stop(1)
    # print('The end')

def trip():
    cap = Picamera2()
    cap.preview_configuration.main.size = (width, height)
    cap.preview_configuration.main.format = "RGB888"
    cap.preview_configuration.align()
    cap.configure("preview")
    cap.start()

    blackFrames = 0
    count_nyStop = [0, 0]
    while True:
        # motor.move(0.7, 0, 0.3)
        # motor.stop(2)

        img = cap.capture_array()
        img = cv2.resize(img, (width, height))
        cv2.imshow('trip', img)

        perceivedW = utils.stopDetector(img, './selfDrivingCar/cascade.xml', 1900)
        count_nyStop[bool(perceivedW)] += 1

        if count_nyStop[1] > 7:

            count_nyStop = [0,0]
            distance3D = utils.distance_to_camera(perceivedW)

            if 27 < distance3D < 40: # in cm
                motor.move(0.3, 0, 0.3)
                motor.stop(2)

        elif count_nyStop[0] > 15:
                count_nyStop = [0,0]

        dist, isEnded = Rm.getLaneCurve(img, display=0)

        if isEnded: blackFrames += 1 
        else: blackFrames = 0
        
        if blackFrames > 8: break

        motor.move(0.3,-dist,0.01)

        if cv2.waitKey(1) == ord("q"):
            break

def turnAround():
    motor.move(0.3, 0, 0.42)
    motor.stop()

    cap = Picamera2()
    cap.preview_configuration.main.size = (width, height)
    cap.preview_configuration.main.format = "RGB888"
    cap.preview_configuration.align()
    cap.configure("preview")
    cap.start()
    motor.parkour()

    while True:
        img = cap.capture_array()
        img = cv2.resize(img,(width,height))
        imgThres = utils.thresholding(img)

        hT, wT = img.shape[:2]
        points = np.float32([(106, 111), (480-106, 111), (24 , 223), (480-24, 223)])
        eagleView = utils.warpImg(imgThres,points,wT,hT)

        if not Rm.RoadEnded(eagleView,0.3):
            motor.stop()
            break

if __name__ == '__main__':
    #main()
    motor.stop()
    