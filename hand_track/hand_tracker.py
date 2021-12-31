import cv2
import mediapipe as mp
import time # check the frame rate

# ADD THIS TO USE MODULE
import HandTrackingModule as hand_track

pTime = 0
cTime = 0

# video capture using webcam 0
cap = cv2.VideoCapture(0)

# ADD THE MODULE SHORTCUT NAME
detector = hand_track.handDetector()

while True:
    success, image = cap.read()

    # if you want drawings to be gone then set "draw = False" to both below
    image = detector.find_hands(image)
    lan_mark_l = detector.find_position(image)

    # for the values at the thumb, when hand is found
    if len(lan_mark_l) != 0:
        print(lan_mark_l[4])

    # get the fps and display it onto the screen
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", image)  # create the window to show the cam
    cv2.waitKey(1)  # delay