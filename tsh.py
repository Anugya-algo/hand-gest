
import cv2 as cv
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


capture = cv.VideoCapture(0)

if not capture.isOpened():
    print("Unable to open web cam")
    exit()


i = 1
while True:
    ret, frame = capture.read()

    if not ret:
        print("Unable to capture frame")
        break


    frame = cv.flip(frame, 1)

    cv.imshow("Camera", frame)


    if cv.waitKey(1) & 0xFF == ord("c"):
        cv.imwrite(f"./left/{i}.png", frame)
        i += 1

    if cv.waitKey(1) & 0xFF == ord("q"):
        print("Closing windows...")
        break
    

cv.destroyAllWindows()
capture.release()
