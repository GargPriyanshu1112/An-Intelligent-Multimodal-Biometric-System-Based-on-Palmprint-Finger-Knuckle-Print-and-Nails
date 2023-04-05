# Import dependencies
import numpy as np
import math
import imutils
import mediapipe as mp
import tensorflow as tf


## **********************  UTILITIES FOR PALMPRINT ROI EXTRACTION  *********************** ##

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



def segment_palmprint_roi(rotated_image, x0, y0, x9, y9, window_w=224, window_h=224):    
    y = y0 + ((y9-y0) / 2)
    x = x9
    
    roi = rotated_image[int(y-(window_h/2)): int(y+(window_h/2)),
                        int(x-(window_w/2)): int(x+(window_w/2)),
                        :]    
    
    return roi



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
            if not abs(x0 - x9) == 0:
                image = rotate_image(image, x0, y0, x9, y9)
            
            # Segment the palmprint region of interest
            roi = segment_palmprint_roi(image, x0, y0, x9, y9)
            
            return roi

## *************************************************************************************** ##






## **************  UTILITIES FOR FINGER NAILS & KNUCKLES ROI EXTRACTION  ***************** ##

mp_hands = mp.solutions.hands
def get_upper_hand_surface_ROIs(image, confidence=0.8):
    """
    Returns ROI of all the landmarks
    """
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
                roi = tf.image.resize(roi, size=(224, 224))
                
                rois.append(roi)
                
            return rois

        

mp_hands = mp.solutions.hands
def get_landmark_ROI(image, i, confidence=0.7, n_h=224, n_w=224):
    """
    Returns ROI of the i-th landmark
    """
    with mp_hands.Hands(min_detection_confidence=confidence) as hands:
        image.flags.writeable = False   
        results = hands.process(image)
        image.flags.writeable = True
        
        # If the model detects hand landmarks
        if results.multi_hand_landmarks:
            # Get the landmarks (normalized)
            landmarks = results.multi_hand_landmarks[0]
            
            scaling_factor = [image.shape[1], image.shape[0]]
            
            # Extract coordinates of i-th landmark
            normalized_coordinates = np.array((landmarks.landmark[i].x, landmarks.landmark[i].y))
            x, y = tuple(np.multiply(normalized_coordinates, scaling_factor).astype(int))
            
            # If x and y coordinates of the landmark are such that complete
            # roi can't be extracted (possible when the bounding box falls out
            # of the image), then skip the image
            if (x < 80 or x > image.shape[1]-80) or (y < 92 or y > image.shape[0]-92):
                return None
            
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
            
            # Resize the roi
            roi = tf.image.resize(roi, size=(n_h, n_w))
            
            return roi
    
## *************************************************************************************** ##
