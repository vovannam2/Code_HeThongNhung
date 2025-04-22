import cv2
import numpy as np
import motors as mot
from picamera2 import Picamera2


def findCenter(p1,p2):
    center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    return center

def minmax_centerPoints(tergetList, pos):
    if len(tergetList) > 0:
        maximum = max(tergetList, key = lambda i: i[pos])
        minimum = min(tergetList, key = lambda i: i[pos])
        return [maximum,minimum]
    return None
    
def detectedlane(imageFrame, width, height):
    
    rotate = 85
    pts1 = [[0, height], [width, height], [width-rotate, rotate], [rotate, rotate]]
    pts2 = [[0, height], [width, height], [width, 0], [0, 0]]

    src = np.float32(pts1)
    des = np.float32(pts2)

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(src, des)
    result = cv2.warpPerspective(frame, matrix, (width, height))
    # cv2.imshow('Result', result)
    
    # cv2.line(imageFrame, (pts1[0][0], pts1[0][1]), (pts1[1][0], pts1[1][1]), (0, 255, 0), 1)
    # cv2.line(imageFrame, (pts1[1][0], pts1[1][1]), (pts1[2][0], pts1[2][1]), (0, 255, 0), 1)
    # cv2.line(imageFrame, (pts1[2][0], pts1[2][1]), (pts1[3][0], pts1[3][1]), (0, 255, 0), 1)
    # cv2.line(imageFrame, (pts1[3][0], pts1[3][1]), (pts1[0][0], pts1[0][1]), (0, 255, 0), 1)
    cv2.imshow('Main Image Window', imageFrame)    
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    threshold = cv2.inRange(gray, 80, 200)
    edges = cv2.Canny(gray, 1, 100, apertureSize=3)
    mergedImage = cv2.add(threshold, edges)
    
    firstSquareCenters1 = findCenter((pts2[1][0], pts2[1][1]), (pts2[2][0], pts2[2][1]))
    firstSquareCenters2 = findCenter((pts2[3][0], pts2[3][1]), (pts2[0][0], pts2[0][1]))
    #cv2.line(result, firstSquareCenters1, firstSquareCenters2, (0,255,0), 1)
    #print("Centers:", firstSquareCenters1, firstSquareCenters2)

    mainFrameCenter = findCenter(firstSquareCenters1, firstSquareCenters2)
    #lines = cv2.HoughLinesP(mergedImage, 1, np.pi/180, 50, minLineLength=80, maxLineGap=50)
    #lines = cv2.HoughLinesP(mergedImage, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    lines = cv2.HoughLinesP(mergedImage, 1, np.pi/180, 50, minLineLength=120, maxLineGap=250)
    centerPoints = []
    left = []
    right = []

    center1 = 0
    center2 = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if 0 <= x1 <= width and 0 <= x2 <= width:
                center = findCenter((x1,y1), (x2,y2))
                if center[0] < (width//2):
                    center1 = center
                    left.append((x1,y1))
                    left.append((x2,y2))
                else:
                    center2 = center
                    right.append((x1,y1))
                    right.append((x2,y2))
                if center1 != 0 and center2 != 0:
                    centroid1 = findCenter(center1, center2)
                    centerPoints.append(centroid1)   

        if len(left) == 0: 
            return 'left'
        if len(right) == 0:
            return 'right'
        
        print(len(left), len(right))   

        centers = minmax_centerPoints(centerPoints, 1)
        laneCenters = 0
        mainCenterPosition = 0        

        if centers is not None:
            laneframeCenter = findCenter(centers[0], centers[1])        
            mainCenterPosition = mainFrameCenter[0] - laneframeCenter[0]            

            cv2.line(result, centers[0], centers[1], [0, 255, 0], 2)
            laneCenters = centers

        return [laneCenters, result, mainCenterPosition]
    
if __name__ == '__main__':
    width, height = 320, 240
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (width, height)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    speed = 0
    maincenter = 0

    while True:
        frame = picam2.capture_array()
        detect = detectedlane(frame, width, height)

        if detect is not None:
            
            if detect == 'left':
                print(detect)
            elif detect == 'right':
                print(detect)
            else:
                maincenter = detect[2]
                print(maincenter)
                
                cv2.putText(detect[1], "Pos=" + str(maincenter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
                cv2.imshow('FinalWindow', detect[1])

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    mot.stop()