# Import dependencies
import mediapipe as mp
import cv2
import os
import random
import matplotlib.pyplot as plt
import numpy as np


# Set up path
dataset_dirpath = "E:/college_project/dataset"


def get_random_img(dirpath):
    # Get random filename for the dataset directory
    fname = random.choice(os.listdir(dirpath))
    # Get filepath
    fpath = os.path.join(dirpath, fname)
    # Read the file
    file = plt.imread(fpath)

    return file


# Get random imagefile
image = get_random_img(dataset_dirpath)

   


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def get_upper_hand_surface_ROIs(image, confidence=0.8):
    rois = [] # will store all the segmented region of interests
    
    with mp_hands.Hands(min_detection_confidence=confidence) as hands:
        image.flags.writeable = False   
        results = hands.process(image)
        image.flags.writeable = True
        
        # If the model detects hand landmarks
        if results.multi_hand_landmarks:
            # Get the landmarks
            landmarks = results.multi_hand_landmarks[0]
            
            # Loop through each landmark
            for i in range(2, 21):
                scale_factor = [image.shape[1], image.shape[0]]
                coordinates = np.array((landmarks.landmark[i].x, landmarks.landmark[i].y))
                
                # Extract Coordinates
                x, y = tuple(np.multiply(coordinates, scale_factor).astype(int))
                
                # Get the dimensions(height & width) of the region of interest
                if i == '2' or '5' or '9' or '13' or '17':
                    window_w, window_h = 150, 168
                elif i == '3' or '6' or '10' or '14' or '18':
                    window_w, window_h = 150, 160
                elif i == '7' or '11' or '15' or '19':
                    window_w, window_h = 150, 140
                elif i == '4' or '8' or '12' or '16' or '20':
                    window_w, window_h = 160,  184
                    
                # Segment the region of interest
                roi = image[int(y-(window_h/2)): int(y+(window_h/2)),
                            int(x-(window_w/2)): int(x+(window_w/2)),
                            :] 
                
                rois.append(roi)
                
            return rois

    
rois = get_upper_hand_surface_ROIs(image)

for idx, roi in enumerate(rois):
    plt.subplot(5, 4, idx+1)
    plt.imshow(roi)
    plt.axis(False)


# results.multi_hand_landmarks