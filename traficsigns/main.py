import cv2
import numpy as np
from picamera2 import Picamera2

sign_detector = cv2.CascadeClassifier("cascade_stop_sign.xml")


    # Dùng để kiểm thử từ hình ảnh
    # img = cv2.imread('Final-2/test-image-1.jpg')
    # img = cv2.resize(img, (WIDTH, HEIGHT))

    # Dùng để kiểm thử từ video
    # cap = cv2.VideoCapture('Final-2/output.mp4')

    # Dùng để lấy ảnh từ điên thoại
    #cap = cv2.VideoCapture('rtsp://192.168.8.241:8080/h264.sdp')
    # ret, img = cap.read()
    
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
    
    
#!/usr/bin/env python

import RPi.GPIO as gpio
from gpiozero import Buzzer
import time
import sys
import signal

def signal_handler(signal, frame):  # ctrl + c -> exit program
    print('You pressed Ctrl+C!')
    gpio.cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

gpio.setmode(gpio.BCM)
trig = 19  # 7th
echo = 26  # 6th

gpio.setup(trig, gpio.OUT)
gpio.setup(echo, gpio.IN)

buzzer = Buzzer(17)

time.sleep(0.5)
print('-----------------------------------------------------------------sonar start')

def measure_distance():
    gpio.output(trig, False)
    time.sleep(0.1)
    gpio.output(trig, True)
    time.sleep(0.00001)
    gpio.output(trig, False)
    while gpio.input(echo) == 0:
        pulse_start = time.time()
    while gpio.input(echo) == 1:
        pulse_end = time.time()
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17000
    if pulse_duration >= 0.01746:
        print('time out')
        return None
    elif distance > 300 or distance == 0:
        print('out of range')
        return None
    distance = round(distance, 3)
    print('Distance : %f cm' % distance)
    return distance

def activate_buzzer():
    buzzer.on()
    time.sleep(0.1)
    buzzer.off()
    time.sleep(0.1)

def main():
    try:
        while True:
            distance = measure_distance()
            if distance is not None and distance < 20:
                activate_buzzer()

    except (KeyboardInterrupt, SystemExit):
        gpio.cleanup()
        sys.exit(0)

    except Exception as e:
        print("Error:", e)
        gpio.cleanup()

if __name__ == "__main__":
    main()




while True:
    width, height = 480, 240
    cap = Picamera2()
    cap.preview_configuration.main.size = (width, height)
    cap.preview_configuration.main.format = "RGB888"
    cap.preview_configuration.align()
    cap.configure("preview")
    cap.start()
    while True: 
        frame = cap.capture_array()
        frame = cv2.resize(frame, (width, height))
        signs = sign_detector.detectMultiScale(frame, 1.3, 5)
        for x, y, w, h in signs:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('FRAME',frame)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
