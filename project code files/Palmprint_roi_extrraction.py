# Import dependencies
import mediapipe as mp
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import imutils
import cv2


# Set up path
dataset_dirpath = "E:/college_project/dataset"


def get_random_img(dirpath):
    fname = random.choice(os.listdir(dirpath))
    fpath = os.path.join(dirpath, fname)
    file = plt.imread(fpath)
    
    return file


# Get random imagefile
image = get_random_img(dataset_dirpath)
lmarks_img = np.copy(image) # create a copy of the image
# plt.imshow(image)
# plt.axis(False)
## ---------------------  ##  ----------------------------  ##  ------------------------  ##

  
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(min_detection_confidence=0.7) as hands:
    image.flags.writeable = False   
    results = hands.process(image)
    image.flags.writeable = True
    
    # If the model detects hand landmarks, loop through each landmark list
    if results.multi_hand_landmarks:
        for _, landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(lmarks_img, landmarks, None,
                                      mp_drawing.DrawingSpec(color=(255,0,0), thickness=20, circle_radius=4))
  
# plt.imshow(lmarks_img)
# plt.axis(False)
"""
TO BE IMPLEMENTED
    If no landmarks are detected, jump to the next image
"""
## ---------------------  ##  ----------------------------  ##  ------------------------  ##


scale = [image.shape[1], image.shape[0]]

coordinates = np.array((landmarks.landmark[0].x, landmarks.landmark[0].y))
X0, Y0 = np.multiply(coordinates, scale).astype(int) # co-ordinates for the loc 0 
Y0 = np.maximum(0, Y0) # if Y0 is -ve, assign Y0 to 0

coordinates = np.array((landmarks.landmark[9].x, landmarks.landmark[9].y))
X9, Y9 = np.multiply(coordinates, scale).astype(int) # co-ordinates for the loc 9 

# plt.imshow(image)
# plt.axis(False)
# plt.plot([X0, X9], [Y0, Y9], marker="o", markersize=5, color="black")
## ---------------------  ##  ----------------------------  ##  ------------------------  ##


def rotate_image(image, x1, y1, x2, y2, check=False):
    # To check if the image is rotated correctly
    if check:
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 8)
    
    y1 = image.shape[0] - y1
    y2 = image.shape[0] - y2
    
    m1 = (y1-y2) / (x1-x2) # slope of line connecting P1:(x1, y1) and P2:(x2, y2)
    
    """
    Using the formula:  tan(theta) = (m1 - m2) / (1 + m1*m2),
    find `theta`. Then subtract theta with 90 degrees in order to obtain angle
    between the horizontal axis and the line joining P1 and P2. This is the angle
    with which the image is to be rotated in order to align the points P1 and P2.
    """
    if m1 > 0:
        angle = 90 - (math.atan(m1) * 180/np.pi)
        rotated_image = imutils.rotate(image, angle) # rotate anti-clockwise
    else:
        angle = 90 + (math.atan(m1) * 180/np.pi)
        rotated_image = imutils.rotate(image, -angle) # rotate clockwise
    
    return rotated_image
    

# Get the rotated image    
image_copy = np.copy(image) # create a copy of the image
rotated_image = rotate_image(image_copy, X0, Y0, X9, Y9)
# plt.imshow(rotated_image)
# plt.axis(False)

# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.plot([X0, X9], [Y0, Y9], marker="o", markersize=5, color="black")
# plt.subplot(1, 2, 2)
# plt.imshow(rotated_image)

## ---------------------  ##  ----------------------------  ##  ------------------------  ##



def pamprint_ROI(rotated_image, X0, Y0, X9, Y9):
    window_w = 250 
    window_h = 250
    
    y = Y0 + ((Y9-Y0) / 2)
    x = X9
    palmprint_bb = cv2.rectangle(np.copy(rotated_image), 
                                 pt1=(int(x-(window_w/2)), int(y-(window_h/2))),  # co-ordinates of top-left corner
                                 pt2=(int(x+(window_w/2)), int(y+(window_h/2))),  # co-ordinates of bottom-right corner
                                 color=(255, 0, 0),
                                 thickness=10)
    
    palmprint_region = rotated_image[int(y-(window_h/2)): int(y+(window_h/2)),
                                     int(x-(window_w/2)): int(x+(window_w/2)),
                                     :]    
    
    return palmprint_bb, palmprint_region, x, y


palmprint_bb, palmprint_region, x, y = pamprint_ROI(rotated_image, X0, Y0, X9, Y9)
# plt.imshow(palmprint_bb)
# plt.axis(False)
# plt.imshow(palmprint_region)
# plt.axis(False)
## ---------------------  ##  ----------------------------  ##  ------------------------  ##




plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis(False)

plt.subplot(1, 3, 2)
plt.imshow(palmprint_bb)
plt.title("Location of ROI")
plt.axis(False)

plt.subplot(1, 3, 3)
plt.imshow(palmprint_region)
plt.title("Extracted palmprint region")
plt.axis(False)







# plt.subplot(1, 3, 1)
# plt.imshow(lmarks_img)
# plt.subplot(1, 3, 2)
# plt.imshow(image)
# plt.plot([X0, X9], [Y0, Y9], marker="o", markersize=5, color="black")
# plt.subplot(1, 3, 3)
# plt.imshow(new_img)
# plt.plot(x, y, marker="o", markersize=5, color="black")


# plt.imshow(rotate_image)
# plt.plot(x, y, marker="o", markersize=5, color="black")
    
