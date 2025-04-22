
import numpy as np
import argparse
import imutils
import cv2
from detection import Detection

sign_detect = Detection()
stream = cv2.VideoCapture(0)
fps = FPS().start()

while True:
	
	(grabbed, frame) = stream.read()
	if not grabbed:
		break
	sign_detect.signDetected(frame)
	cv2.imshow("Frame", frame)
	if cv2.waitKey(10) == 27:
                break
	fps.update()

fps.stop()
stream.release()
cv2.destroyAllWindows()
