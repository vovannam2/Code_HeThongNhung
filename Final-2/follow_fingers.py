import cv2
import numpy as np
import mediapipe as mp
from MotorModule import Motor
from picamera2 import Picamera2

# Initialize MediaPipe Hands solution
mp_drawing_util = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_hand = mp.solutions.hands

# Initialize Hands
hands = mp_hand.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

WIDTH, HEIGHT = 480, 240
IMGCENTER = WIDTH // 2    

# Function to process the frame and return the deviation
def process_frame(frame):
    h, w, _ = frame.shape
    center_x = w // 2

    # Convert the image to RGB before processing
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    deviation = None

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_drawing_util.draw_landmarks(
                frame,
                hand,
                mp_hand.HAND_CONNECTIONS,
                mp_drawing_style.get_default_hand_landmarks_style(),
                mp_drawing_style.get_default_hand_connections_style()
            )

            # Extract the index fingertip landmark (ID 8)
            index_fingertip = hand.landmark[8]
            fingertip_x = int(index_fingertip.x * w)
            fingertip_y = int(index_fingertip.y * h)

            # Calculate deviation from the center
            deviation = fingertip_x - center_x

            # Draw the vertical center line
            cv2.line(frame, (center_x, 0), (center_x, h), (0, 255, 0), 2)

    return frame, deviation

def smoothed(dist):
    normalizedDist = dist / IMGCENTER
    n = np.sqrt(np.log10(abs(normalizedDist)/4+1))
    return n * (dist > 15) - n * (dist < -15)

if __name__ == '__main__':

    # Khai báo các chân kết nối đến motor
    motor = Motor(25,23,24,16,21,20)

    # Dùng để trong raspberry
    cap = Picamera2()
    cap.preview_configuration.main.size = (WIDTH, HEIGHT)
    cap.preview_configuration.main.format = "RGB888"
    cap.preview_configuration.controls.FrameRate = 30
    cap.preview_configuration.align()
    cap.configure("preview")
    cap.start()

    # Dùng để kiểm thử từ video webcam
    # cap = cv2.VideoCapture(0)
    
    while True:  
        success, img = cap.read()
        img = cv2.resize(img, (WIDTH, HEIGHT))

        if success:
            img, deviation = process_frame(img)
            if deviation is not None and abs(deviation) > 0.0:
                cureve = smoothed(deviation)
                #motor.move(0.27, cureve)

                print(cureve)       
                #cv2.putText(img, f'Deviation: {deviation}px', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                #cv2.imshow('Finger Deviation Detection', img)
            
        if cv2.waitKey(1) == ord("q"):
            motor.stop()
            break

    cv2.destroyAllWindows()

