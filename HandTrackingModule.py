import cv2
import mediapipe as mp
import time # check the frame rate


class handDetector():
    def __init__(self, mode = False, max_hands = 2, detection_confidence = 0.5, track_confidence = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.mpHands = mp.solutions.hands
        # default params from Hands (depending on confidence, detect or track,2 hands)
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.detection_confidence, self.track_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, image, draw = True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # get the video in color
        self.results = self.hands.process(imageRGB)  # processes the hand in the RGB video

        if self.results.multi_hand_landmarks:
            for handMark in self.results.multi_hand_landmarks:
                if draw:
                    # show each hand with landmarks in the original video also with connections
                    self.mpDraw.draw_landmarks(image, handMark, self.mpHands.HAND_CONNECTIONS)
        return image

    def find_position(self, image, hand_num = 0, draw = True):
        land_mark_l = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_num]

            for id, lm in enumerate(my_hand.landmark):  # location of the point
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                land_mark_l.append([id, cx, cy])

                if draw:
                    cv2.circle(image, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
        return land_mark_l

def main():
    pTime = 0
    cTime = 0

    # video capture using webcam 0
    cap = cv2.VideoCapture(0)

    detector = handDetector()

    while True:
        success, image = cap.read()

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


if __name__ == "__main__":
    main()
