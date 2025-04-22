import cv2
import numpy as np
import utils
from MotorModule import Motor
from picamera2 import Picamera2
import HCSR04 as hcs
#import stop_sign_detection as stop

def empty(h):
    pass


stop_sign = cv2.CascadeClassifier('cascade_stop_sign.xml')

def detect_stop_sign(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stop_sign_scaled = stop_sign.detectMultiScale(gray, 1.3, 5)

    if len(stop_sign_scaled) > 0:  # Nếu có biển báo stop được phát hiện
        for (x, y, w, h) in stop_sign_scaled:
            # Vẽ khung xanh xung quanh biển báo
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            # Thêm văn bản "Stop Sign" dưới biển báo
            img = cv2.putText(img, "Stop Sign", (x, y+h+30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        return "stop"
    else:
        return "go"


def getLaneCurve(img):
    # Tạo mask tách nền
    imgBorder = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    imgHsv = cv2.cvtColor(imgBorder, cv2.COLOR_BGR2HSV)


    # Hiệu chỉnh màu
    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VAL Min", "HSV")
    v_max = cv2.getTrackbarPos("VAL Max", "HSV")
    lowerWhite = np.array([h_min, s_min, v_min])
    upperWhite = np.array([h_max, s_max, v_max])

    lowerWhite = np.array([0, 0, 0])  
    upperWhite = np.array([179, 255, 115])
    maskWhite = cv2.inRange(imgHsv, lowerWhite, upperWhite)
    
     # Wrap hình ảnh
    hT, wT = img.shape[:2]
    wrapYTopLeft = 20
    wrapYTopRight = 20
    wrapXTopLeft = 110
    wrapXTopRight = 70

    pts1 = [[0, height], [width, height], [width-wrapXTopRight, wrapYTopLeft], [wrapXTopLeft, wrapYTopRight]]
    pts2 = [[0, height], [width, height], [width, 0], [0, 0]]
    
    src = np.float32(pts1)
    des = np.float32(pts2)

    matrix = cv2.getPerspectiveTransform(src, des)
    imgWarp = cv2.warpPerspective(maskWhite, matrix, (wT, hT))

    # Tính Histogram
    minPer = 0.5
    region = 4

    h, w = imgWarp.shape[:2]
    histValues = np.sum(imgWarp[-h//region:, :], axis=0)

    maxValue = np.max(histValues)
    minValue = minPer * maxValue

    indexArray = np.where(histValues >= minValue)
    RoadCenter = int(np.average(indexArray))
    imgHist = np.zeros((h, w, 3), np.uint8)

    for x, intensity in enumerate(histValues):
        if intensity > minValue:color=(211, 211, 211)
        else: color=(200, 165, 200)
        cv2.line(imgHist, (x, h), (x, int(h-(intensity//region//255))), color, 1)
        cv2.circle(imgHist, (RoadCenter, h), 20, (255, 200, 0), cv2.FILLED)
    
    imgCenter = 240    
    dist = (RoadCenter - imgCenter)
    smoothDist = smoothed(dist)
    print(smoothDist)
    imgWarpPoints = img.copy()    
    imgResult = img.copy()
    for x in range(4):
        cv2.circle(imgWarpPoints, (int(pts1[x][0]), int(pts1[x][1])), 15, (255, 0, 0), cv2.FILLED)
    
    display = 2
    if display != 0:
        imgInvWarp = utils.warpImg(imgWarp, pts1, wT, hT, inv = True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:hT//3, 0:wT] = 0, 0, 0
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        smoothImg = imgResult.copy()

        if dist >=0:
            cv2.putText(imgResult, str(dist)[:5], (RoadCenter-15, hT//2), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0), 3)
        else:
            cv2.putText(imgResult, str(dist)[:5], (RoadCenter-40, hT//2), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0), 3)

        cv2.circle(imgResult, (imgCenter, hT-10), 9, (0, 0, 255), cv2.FILLED)
        cv2.circle(imgResult, (RoadCenter, hT-10), 8, (255, 200, 0), 3)
        cv2.line(imgResult, (imgCenter, hT-10), (RoadCenter, hT-10), (0, 0, 0), 4) 

    if display == 2:
        cv2.putText(smoothImg, str(round(smoothDist, 2)), (wT//2-80, hT//2), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0), 3)
        imgStacked = utils.stackImages(1, ([img, maskWhite, imgWarpPoints], [imgWarp, imgHist, imgResult]))
        cv2.imshow('ImageStack', imgStacked)

    return smoothDist


def smoothed(dist):
    normalizedDist = dist/240
    n = np.sqrt(np.log10(abs(normalizedDist)/4+1))
    return n*(dist>15)-n*(dist<-15)

if __name__ == '__main__':
    motor = Motor(25,23,24,16,21,20)
    width, height = 480, 240

    piCam = Picamera2()
    piCam.preview_configuration.main.size=(480,240)
    piCam.preview_configuration.main.format='RGB888'
    piCam.preview_configuration.controls.FrameRate=60
    piCam.preview_configuration.align()
    piCam.configure('preview')
    piCam.start()
    fps=0

    while True:  
        img = piCam.capture_array()
        result = detect_stop_sign(img)
        print(result)
        if result == "go":
            curve = getLaneCurve(img)
            print(curve)
            distance = hcs.measure_distance()
            if distance is not None and distance < 20:
                hcs.activate_buzzer()  
                #distance = 10
                #motor.move(0.25, 0.25, 1.5) 
                #hcs.activate_buzzer() 
                #motor.move(0.25, -0.25, 1.5) 
                #hcs.activate_buzzer()
                #motor.move(0.3, 0, 1)
                #motor.move(0.25, -0.25, 1.5) 
                #hcs.activate_buzzer()
                #motor.move(0.25, 0.25, 1.5)
                #hcs.activate_buzzer()
                #distance = 40
                
                distance = hcs.measure_distance()
            if distance is not None and distance < 20:
                distance = 10
                hcs.activate_buzzer()
                motor.move(-0.5, 0, 0.3)
                motor.move(0.3, 0.5, 0.4)
                motor.move(0.5, 0, 0.7)
                motor.move(0.3, -0.5, 0.4)
                motor.move(0.3, -0.5, 0.3)
                motor.move(0.5, 0, 0.8)
                motor.move(0.2, 0.5, 0.3)
                motor.move(0.5, 0, 0.5)
                distance = 40
            motor.move(0.2, curve)
            distance = hcs.measure_distance()
            if distance is not None and distance < 25:
                distance = 10
                hcs.activate_buzzer()
                motor.move(-0.5, 0, 0.3)
                motor.move(0.3, 0.5, 0.4)
                motor.move(0.5, 0, 0.7)
                motor.move(0.3, -0.5, 0.4)
                motor.move(0.3, -0.5, 0.3)
                motor.move(0.5, 0, 0.8)
                motor.move(0.2, 0.5, 0.3)
                motor.move(0.5, 0, 0.5)
                distance = 40
        else:
            motor.stop()
        cv2.imshow("Frame", img)
        if cv2.waitKey(1) == ord("q"):
            motor.stop()
            break



