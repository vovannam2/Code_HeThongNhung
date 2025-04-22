import cv2
import numpy as np
import utils
from MotorModule import Motor
from picamera2 import Picamera2
import HCSR04 as hcs

WIDTH, HEIGHT = 480, 240
IMGCENTER = WIDTH // 2    

def empty(h): pass

def getLaneCurve(img):
    # Tạo mask tách nền
    imgBorder = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    imgHsv = cv2.cvtColor(imgBorder, cv2.COLOR_BGR2HSV)


    lowerWhite = np.array([0, 0, 0])
    upperWhite = np.array([179, 255, 115])
    maskWhite = cv2.inRange(imgHsv, lowerWhite, upperWhite)
    
    # Hiệu chỉnh viewbird
    YTopLeft = cv2.getTrackbarPos("YTopLeft", "Trackbars")
    YTopRight = cv2.getTrackbarPos("YTopRight", "Trackbars")
    XTopLeft = cv2.getTrackbarPos("XTopLeft", "Trackbars")
    XTopRight = cv2.getTrackbarPos("XTopRight", "Trackbars")
    wrapYTopLeft = YTopLeft
    wrapYTopRight = YTopRight
    wrapXTopLeft = XTopLeft
    wrapXTopRight = XTopRight

    # Wrap hình ảnh
    hT, wT = img.shape[:2]
    wrapYTopLeft = 20
    wrapYTopRight = 20
    wrapXTopLeft = 45
    wrapXTopRight = 90

    pts1 = [[0, HEIGHT], [WIDTH, HEIGHT], [WIDTH-wrapXTopRight, wrapYTopRight], [wrapXTopLeft, wrapYTopLeft]]
    pts2 = [[0, HEIGHT], [WIDTH, HEIGHT], [WIDTH, 0], [0, 0]]

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
        else: color = (200, 165, 200)
        cv2.line(imgHist, (x, h), (x, int(h-(intensity//region//255))), color, 1)
        cv2.circle(imgHist, (RoadCenter, h), 20, (255, 200, 0), cv2.FILLED)
    
    dist = (RoadCenter - IMGCENTER)
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

        cv2.circle(imgResult, (IMGCENTER, hT-10), 9, (0, 0, 255), cv2.FILLED)
        cv2.circle(imgResult, (RoadCenter, hT-10), 8, (255, 200, 0), 3)
        cv2.line(imgResult, (IMGCENTER, hT-10), (RoadCenter, hT-10), (0, 0, 0), 4) 

    if display == 2:
        cv2.putText(smoothImg, str(round(smoothDist, 2)), (wT//2-80, hT//2), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0), 3)
        imgStacked = utils.stackImages(1, ([img, maskWhite, imgWarpPoints], [imgWarp, imgHist, imgResult]))
        cv2.imshow('ImageStack', imgStacked)

    return smoothDist


def smoothed(dist):
    normalizedDist = dist / IMGCENTER
    n = np.sqrt(np.log10(abs(normalizedDist)/4+1))
    return n * (dist > 15) - n * (dist < -15)

turnLeft_ahead = cv2.CascadeClassifier('turnLeft_ahead.xml')
turnRight_ahead = cv2.CascadeClassifier('turnRight_ahead.xml')
stop_sign = cv2.CascadeClassifier('cascade_stop_sign.xml')
dibo_sign = cv2.CascadeClassifier('dibosign.xml')

def detect_sign(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stop_sign_scaled = 0
    turnRight = 0
    turnLeft = 0
    dibo = 0
    stop_sign_scaled = stop_sign.detectMultiScale(gray, 1.3, 5)
    turnRight = turnRight_ahead.detectMultiScale(gray, 1.3, 5)
    turnLeft = turnLeft_ahead.detectMultiScale(gray, 1.3, 5)
    dibo = dibo_sign.detectMultiScale(gray, 1.3, 5)
    
    if (len(stop_sign_scaled) > 0):  
        return "stop"
    if (len(turnLeft)) > 0:  
        return "left"
    if (len(turnRight)) > 0:  
        return "right"
    if (len(dibo)) > 0:  
        return "dibo"
    return "go"
temp = []
def scaleSpeed(x):
    temp.append(x)
    m = np.std(temp)

    if len(temp) == 10 and m < 0.1:
        temp.clear()
        return 1.2

    return 1

if __name__ == '__main__':

    # Khai báo các chân kết nối đến motor
    motor = Motor(25,23,24,16,21,20)

    # Dùng để trong raspberry
    cap = Picamera2()
    cap.preview_configuration.main.size = (WIDTH, HEIGHT)
    cap.preview_configuration.main.format = "RGB888"
    cap.preview_configuration.controls.FrameRate = 60
    cap.preview_configuration.align()
    cap.configure("preview")
    cap.start()

    # Dùng để kiểm thử từ hình ảnh
    # img = cv2.imread('Final-2/test-image-1.jpg')
    # img = cv2.resize(img, (WIDTH, HEIGHT))

    # Dùng để kiểm thử từ video
    # cap = cv2.VideoCapture('Final-2/output.mp4')

    # Dùng để lấy ảnh từ điên thoại
    # cap = cv2.VideoCapture('rtsp://192.168.8.241:8080/h264.sdp')

    
    while True:  
        imgOrignal = cap.capture_array()
        img = cv2.resize(imgOrignal, (WIDTH, HEIGHT))
        result = "go"
        result = detect_sign(img)
        print(result)
        if (result == "left"):
            motor.move(0.3, -0.5, 0.4)
            motor.move(0.3, -0.5, 0.4)
            motor.move(0.3, 0, 0.5)
        if (result == "right"):
            motor.move(0.3, 0.5, 0.4)
            motor.move(0.3, 0.5, 0.4)
            motor.move(0.3, 0, 0.3)                
        if (result == "dibo"):
            hcs.activate_buzzer()
            motor.stop(3)

        if cv2.waitKey(1) == ord("q"):
            motor.stop()
            break

    cv2.destroyAllWindows()

