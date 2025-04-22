# Python Script
# https://www.electronicshub.org/raspberry-pi-l298n-interface-tutorial-control-dc-motor-l298n-raspberry-pi/

import RPi.GPIO as GPIO          
from time import sleep

inl1 = 24
inl2 = 23
inr1 = 20
inr2 = 21

enl = 25
enr = 16

temp1=1

GPIO.setmode(GPIO.BCM)
GPIO.setup(inl1,GPIO.OUT)
GPIO.setup(inl2,GPIO.OUT)
GPIO.setup(inr1,GPIO.OUT)
GPIO.setup(inr2,GPIO.OUT)

GPIO.setup(enl,GPIO.OUT)
GPIO.setup(enr,GPIO.OUT)

GPIO.output(inl1,GPIO.LOW)
GPIO.output(inl2,GPIO.LOW)
GPIO.output(inr1,GPIO.LOW)
GPIO.output(inr2,GPIO.LOW)

pl=GPIO.PWM(enl,1000)
pr=GPIO.PWM(enr,1000)

pl.start(75)
pr.start(75)

GPIO.output(inl1,GPIO.HIGH)
GPIO.output(inl2,GPIO.LOW)
GPIO.output(inr1,GPIO.HIGH)
GPIO.output(inr2,GPIO.LOW)
