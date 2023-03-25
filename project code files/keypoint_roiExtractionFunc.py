# Import dependencies
import mediapipe as mp
import cv2
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


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

# img_fpath = os.path.join(dataset_dirpath, 'Hand_0001795.jpg')
# image = plt.imread(img_fpath)

   


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
            
            scale_factor = [image.shape[1], image.shape[0]]
            
            # Loop through each landmark
            for i in range(2, 21):
                # Extract coordinates of i-th keypoint
                normalized_coordinates = np.array((landmarks.landmark[i].x, landmarks.landmark[i].y))
                x, y = tuple(np.multiply(normalized_coordinates, scale_factor).astype(int))
                
                # If x and y coordinates of a landmark are such that complete roi
                # can't be extracted (possible when the bounding box falls out
                # of the image), then skip the landmark
                if (x < 80 or x > image.shape[1]-80) or (y < 92 or y > image.shape[0]-92):
                    continue
                
                # Get the dimensions(height & width) of the roi
                if i in [2, 5, 9, 13, 17]:
                    window_w, window_h = 150, 168
                    
                elif i in [3, 6, 10, 14, 18]:
                    window_w, window_h = 150, 160

                elif i in [7, 11, 15, 19]:
                    window_w, window_h = 150, 140

                elif i in [4, 8, 12, 16, 20]:
                    window_w, window_h = 160,  184
                    
                # Segment the region of interest
                roi = image[int(y-(window_h/2)): int(y+(window_h/2)),
                            int(x-(window_w/2)): int(x+(window_w/2)),
                            :] 
                
                # Normalize the roi, so the pixel values are between [0, 1]
                roi = roi / 255.0
                
                # Resize the roi to 224 x 224, as the feature extraction 
                # model (DenseNet201) expects a fixed sized input image
                roi = tf.image.resize(roi, size=(224, 224)) #########################
                
                rois.append(roi)
                
            return rois

    
rois = get_upper_hand_surface_ROIs(image)

for idx, roi in enumerate(rois):
    plt.subplot(5, 4, idx+1)
    plt.imshow(roi)
    plt.axis(False)


# results.multi_hand_landmarks
