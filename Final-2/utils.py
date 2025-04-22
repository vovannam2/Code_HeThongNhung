import cv2
import numpy as np

def nothing(a):
    pass

def warpImg(img, points, w, h, inv = False):
    pts1 = np.float32(points)
    pts2 =  np.float32([[0,0],[w,0],[0,h],[w,h]])

    if inv:
        matrix = cv2.getPerspectiveTransform(pts2,pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1,pts2)

    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    return imgWarp

def stackImages(scale, imgArray):
    if isinstance(imgArray[0], list):
        return imageMatrix(scale, imgArray)
    
    return imageArray(scale, imgArray)

def imageMatrix(scale, imgArray):
    rows, cols = len(imgArray), len(imgArray[0])
    hImg, wImg = imgArray[0][0].shape[:2]
    hImg, wImg = hImg//2 , wImg//2
    for x in range(0, rows):
        for y in range(0, cols):
            imgArray[x][y] = cv2.resize(imgArray[x][y], (int(wImg*scale), int(hImg*scale)), None)
            if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)

    imageBlank = np.zeros((hImg, wImg, 3), np.uint8)
    hor = [imageBlank]*rows
    for x in range(0, rows):
        hor[x] = np.hstack(imgArray[x])

    return np.vstack(hor)

def imageArray(scale, imgArray):
    for x in range(0, len(imgArray)):
        imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)

        if len(imgArray[x].shape) == 2: 
            imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)

    return np.hstack(imgArray)

def distance_to_camera(reflectedWidth):
    knownWidth = 4.5 
    focalLength = 330 
    return (knownWidth * focalLength) / reflectedWidth

