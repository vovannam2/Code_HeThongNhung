#MOTOR CLASS
import RPi.GPIO as GPIO
from time import sleep

import RPi.GPIO as GPIO          
from time import sleep

#PIN ben trai
#1 HEGIH -> 2 LOW -> Di thang
inl1 = 24 
inl2 = 23

#PIN ben phai
#1 HEGIH -> 2 LOW -> Di thang
inr1 = 20
inr2 = 21

enl = 25
enr = 16

#Khai bao
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

#Khai bao cycle 
pl=GPIO.PWM(enl,1000)
pr=GPIO.PWM(enr,1000)

#Chon so cycle la 50 vong/s
cycle = 50
pl.start(cycle)
pr.start(cycle)

def stop():
    GPIO.output(inl1,GPIO.LOW)
    GPIO.output(inl2,GPIO.LOW)
    GPIO.output(inr1,GPIO.LOW)
    GPIO.output(inr2,GPIO.LOW)

def forward():
    GPIO.output(inl1,GPIO.HIGH)
    GPIO.output(inl2,GPIO.LOW)
    GPIO.output(inr1,GPIO.HIGH)
    GPIO.output(inr2,GPIO.LOW)

def backward():
    GPIO.output(inl1,GPIO.LOW)
    GPIO.output(inl2,GPIO.HIGH)
    GPIO.output(inr1,GPIO.LOW)
    GPIO.output(inr2,GPIO.HIGH)

def rotate_left():
    GPIO.output(inl1,GPIO.LOW)
    GPIO.output(inl2,GPIO.HIGH)
    GPIO.output(inr1,GPIO.HIGH)
    GPIO.output(inr2,GPIO.LOW)

def rotate_right():
    GPIO.output(inl1,GPIO.HIGH)
    GPIO.output(inl2,GPIO.LOW)
    GPIO.output(inr1,GPIO.LOW)
    GPIO.output(inr2,GPIO.HIGH)
    
def exit():
    GPIO.cleanup()

if __name__ == '__main__':    
    exit()