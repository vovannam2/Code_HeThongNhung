import cv2
import numpy as np
import utils
from MotorModule import Motor
from picamera2 import Picamera2
import HCSR04 as hcs

WIDTH, HEIGHT = 480, 240
IMGCENTER = WIDTH // 2    

def empty(h): pass

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", WIDTH, HEIGHT)
cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
cv2.createTrackbar("HUE Max", "HSV", 179, 179,  empty)
cv2.createTrackbar("SAT Min", "HSV", 0, 255,  empty)
cv2.createTrackbar("SAT Max", "HSV", 150, 255,  empty)
cv2.createTrackbar("VAL Min", "HSV", 0, 255,  empty)
cv2.createTrackbar("VAL Max", "HSV",115, 255,  empty)

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", WIDTH, HEIGHT)
cv2.createTrackbar("XTopLeft", "Trackbars", 75, WIDTH//2, empty)
cv2.createTrackbar("YTopLeft", "Trackbars", 79, HEIGHT, empty)
cv2.createTrackbar("XTopRight", "Trackbars", 53, WIDTH//2, empty)
cv2.createTrackbar("YTopRight", "Trackbars", 77, HEIGHT, empty)

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

    #lowerWhite = np.array([0, 0, 0])
    #upperWhite = np.array([179, 150, 115])
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
    wrapYTopLeft = 20
    wrapYTopRight = 20
    wrapXTopLeft = 110
    wrapXTopRight = 70
    hT, wT = img.shape[:2]

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
    n = np.sqrt(np.log10(1.3*abs(normalizedDist)/4+1))
    return n * (dist > 5) - n * (dist < 5)

stop_sign = cv2.CascadeClassifier('cascade_stop_sign.xml')

def detect_stop_sign(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stop_sign_scaled = stop_sign.detectMultiScale(gray, 1.3, 5)

    if len(stop_sign_scaled) > 0:  # Nếu có biển báo stop được phát hiện
        #for (x, y, w, h) in stop_sign_scaled:
            # Vẽ khung xanh xung quanh biển báo
            #img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            # Thêm văn bản "Stop Sign" dưới biển báo
            #img = cv2.putText(img, "Stop Sign", (x, y+h+30), 
                              #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        return "stop"
    else:
        return "go"


if __name__ == '__main__':

    # Khai báo các chân kết nối đến motor
    motor = Motor(25,23,24,16,21,20)

    # Dùng để trong raspberry
    cap = Picamera2()
    cap.preview_configuration.main.size = (WIDTH, HEIGHT)
    cap.preview_configuration.main.format = "RGB888"
    cap.preview_configuration.controls.FrameRate = 30
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
        result = detect_stop_sign(img)
        print(result)
        if result == "go":
            curve = getLaneCurve(img)
            print(curve)
        
            distance = hcs.measure_distance()
            if distance is not None and distance < 30:
                hcs.activate_buzzer()  
                # code né vật cản
                # motor.move(0.25, 0.25, 1.5)  
                # motor.move(0.25, -0.25, 1.5)                 
                # motor.move(0.25, -0.25, 1.5) 
                # motor.move(0.25, 0.25, 1.5)
                distance = 40 
        
            motor.move(0.25, -curve)
        else:
            motor.stop(3)

        if cv2.waitKey(1) == ord("q"):
            motor.stop()
            break

    cv2.destroyAllWindows()

