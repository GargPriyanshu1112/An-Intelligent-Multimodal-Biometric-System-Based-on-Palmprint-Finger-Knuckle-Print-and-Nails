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


  
"""
TO BE IMPLEMENTED
    If no landmarks are detected, jump to the next image
"""
## ---------------------  ##  ----------------------------  ##  ------------------------  ##

def rotate_image(image, x1, y1, x2, y2):    
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



def segment_palmprint_roi(rotated_image, X0, Y0, X9, Y9):
    window_w = 250 
    window_h = 250
    
    y = Y0 + ((Y9-Y0) / 2)
    x = X9
    
    roi = rotated_image[int(y-(window_h/2)): int(y+(window_h/2)),
                        int(x-(window_w/2)): int(x+(window_w/2)),
                        :]    
    
    return roi



mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def get_inner_hand_surface_ROI(image, confidence=0.8):    
    with mp_hands.Hands(min_detection_confidence=confidence) as hands:
        image.flags.writeable = False   
        results = hands.process(image)
        image.flags.writeable = True
        
        # If the model detects hand landmarks
        if results.multi_hand_landmarks:
            # Get the landmarks
            landmarks = results.multi_hand_landmarks[0]
            
            scaling_factor = [image.shape[1], image.shape[0]]
            
            # Get the coordinates of keypoint-0 (WRIST)
            normalized_coordinates = np.array((landmarks.landmark[0].x, landmarks.landmark[0].y))
            x0, y0 = np.multiply(normalized_coordinates, scaling_factor).astype(int)
            y0 = np.maximum(0, y0) # if y-coordinate is -ve, assign it to 0
            
            # Get the coordinates of keypoint-9 (MIDDLE_FINGER_MCP)
            normalized_coordinates = np.array((landmarks.landmark[9].x, landmarks.landmark[9].y))
            x9, y9 = np.multiply(normalized_coordinates, scaling_factor).astype(int)
            
            # Rotate the image
            rotated_image = rotate_image(image, x0, y0, x9, y9)
            
            # Segment the palmprint region of interest
            roi = segment_palmprint_roi(rotated_image, x0, y0, x9, y9)
            
            return roi, rotated_image


            
roi, rotated_image = get_inner_hand_surface_ROI(image)

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis(False)

plt.subplot(1, 3, 2)
plt.imshow(rotated_image)
plt.title("Rotated Image")
plt.axis(False)

plt.subplot(1, 3, 3)
plt.imshow(roi)
plt.title("Extracted palmprint region")
plt.axis(False)

