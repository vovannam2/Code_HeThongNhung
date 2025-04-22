import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)

while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        print('No frames grabbed!')
        break
    
    key = cv.waitKey(1)
    if key == 27:
        break

    # Visualize results
    cv.imshow('Live', frame)
cv.destroyAllWindows()