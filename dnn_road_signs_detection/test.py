import numpy as np
import cv2 as cv

inp = np.random.standard_normal([1, 3, 256, 320]).astype(np.float32)
net = cv.dnn.readNetFromONNX('/yolov8n_road_signs.onnx')
#net.setInput(inp)
#out = net.forward()
#print(out.shape)