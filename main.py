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

pl.start(25)
pr.start(25)
print("\n")
print("The default speed & direction of motor is LOW & Forward.....")
print("r-run s-stop f-forward b-backward l-low m-medium h-high e-exit")
print("\n")    

while(1):

    x=input()
    
    if x=='r':
        print("run")
        if(temp1==1):
         GPIO.output(inl1,GPIO.HIGH)
         GPIO.output(inl2,GPIO.LOW)
         GPIO.output(inr1,GPIO.HIGH)
         GPIO.output(inr2,GPIO.LOW)
         print("forward")
         x='z'
        else:
         GPIO.output(inl1,GPIO.LOW)
         GPIO.output(inl2,GPIO.HIGH)
         GPIO.output(inr1,GPIO.LOW)
         GPIO.output(inr2,GPIO.HIGH)
         print("backward")
         x='z'


    elif x=='s':
        print("stop")
        GPIO.output(inl1,GPIO.LOW)
        GPIO.output(inl2,GPIO.LOW)
        GPIO.output(inr1,GPIO.LOW)
        GPIO.output(inr2,GPIO.LOW)
        x='z'

    elif x=='f':
        print("forward")
        GPIO.output(inl1,GPIO.HIGH)
        GPIO.output(inl2,GPIO.LOW)
        GPIO.output(inr1,GPIO.HIGH)
        GPIO.output(inr2,GPIO.LOW)
        temp1=1
        x='z'

    elif x=='b':
        print("backward")
        GPIO.output(inl1,GPIO.LOW)
        GPIO.output(inl2,GPIO.HIGH)
        GPIO.output(inr1,GPIO.LOW)
        GPIO.output(inr2,GPIO.HIGH)
        temp1=0
        x='z'
    elif x=='rl':
        print("rotate left")
        GPIO.output(inl1,GPIO.LOW)
        GPIO.output(inl2,GPIO.HIGH)
        GPIO.output(inr1,GPIO.HIGH)
        GPIO.output(inr2,GPIO.LOW)
        pl.ChangeDutyCycle(50)
        pr.ChangeDutyCycle(50)
        
        x='z'
    elif x=='rr':
        print("rotate right")
        GPIO.output(inl1,GPIO.HIGH)
        GPIO.output(inl2,GPIO.LOW)
        GPIO.output(inr1,GPIO.LOW)
        GPIO.output(inr2,GPIO.HIGH)
        pl.ChangeDutyCycle(50)
        pr.ChangeDutyCycle(50)
        
        x='z'
    elif x=='l':
        print("low")
        pl.ChangeDutyCycle(25)
        pr.ChangeDutyCycle(25)
        x='z'

    elif x=='m':
        print("medium")
        pl.ChangeDutyCycle(50)
        pr.ChangeDutyCycle(50)
        x='z'

    elif x=='h':
        print("high")
        pl.ChangeDutyCycle(75)
        pr.ChangeDutyCycle(75)
        x='z'
     
    
    elif x=='e':
        GPIO.cleanup()
        print("GPIO Clean up")
        break
    
    else:
        print("<<<  wrong data  >>>")
        print("please enlter the defined data to continue.....")
