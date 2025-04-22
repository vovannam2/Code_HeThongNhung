from MotorModule import Motor
import KeyPressModule as kp
import cv2

motor = Motor(25,23,24,16,21,20)

kp.init()

def main():
    if kp.getKey('w'):
        motor.move(0.4)
    elif kp.getKey('s'):
        motor.move(-0.4)
    elif kp.getKey('d'):
        motor.move(0.4,-0.3)
    elif kp.getKey('a'):
        motor.move(0.4,0.3)
    elif kp.getKey('q'):
        motor.move(0.25,-0.5)
    elif kp.getKey('e'):
        motor.move(0.25,0.5)
    elif kp.getKey('z'):
        motor.move(0.25,-0.5)
    elif kp.getKey('c'):
        motor.move(0.25,0.5)
    else:
        motor.stop()

if __name__ == '__main__':
    while True:
        main()
        if kp.getKey('p'):
            kp.Quit()
            break
