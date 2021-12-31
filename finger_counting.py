import cv2
import os
import time # check the frame rate

width_cam, height_cam = 640, 480

# video capture using webcam 0
cap = cv2.VideoCapture(0)
# cv2.resizeWindow("Image", 200, 300)


# get the picture of the numbers
folder_path = "num_images"
my_numbers = os.listdir(folder_path)

overlay_numlist = []  # list of numbers in the file path

# get the numbers with their paths
for img_path in my_numbers:
    image = cv2.imread(f'{folder_path}/{img_path}')
    overlay_numlist.append(image)

while True:
    success, image = cap.read()

    image = overlay_numlist[2]

    cv2.imshow("Image", image)  # create the window to show the cam
    cv2.waitKey(1)  # delay