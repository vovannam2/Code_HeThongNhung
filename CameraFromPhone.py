import cv2

cap = cv2.VideoCapture('rtsp://192.168.8.241:8080/h264.sdp')
#cap = cv2.VideoCapture(0)
#cam_pipeline_str = 'nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),format=NV12,width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1'
#cap = cv2.VideoCapture(cam_pipeline_str, cv2.CAP_GSTREAMER)
while True:
    success, imgOrignal = cap.read()
    
    if success == True:
        img = cv2.resize(imgOrignal, (480, 240))
        cv2.imshow("Result", img)

    if cv2.waitKey(1) == ord('q'):
        break