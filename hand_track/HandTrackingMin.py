import cv2
import mediapipe as mp

import time # check the frame rate

# video capture using webcam 0
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands() # default params from Hands (depending on confidence choose to detect or track, max of 2 hands)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # get the video in color
    results = hands.process(imageRGB)  # processes the hand in the RGB video

    if results.multi_hand_landmarks:
        for handMark in results.multi_hand_landmarks:
            for id, lm in enumerate(handMark.landmark):  # location of the point
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)

                # if id == 4:
                cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            # show each hand with landmarks in the original video also with connections
            mpDraw.draw_landmarks(image, handMark, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", image)  # create the window to show the cam
    cv2.waitKey(1)  # delay
