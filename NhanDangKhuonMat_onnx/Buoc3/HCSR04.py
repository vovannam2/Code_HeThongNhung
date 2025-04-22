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

def measure_distance():
    gpio.output(trig, False)
    time.sleep(0.1)
    gpio.output(trig, True)
    time.sleep(0.00001)
    gpio.output(trig, False)
    pulse_start = time.time()
    pulse_end = time.time()
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



