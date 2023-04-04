# Import dependencies
import numpy as np
import cv2
import os
from PIL import Image

# Set up path
DATASET_DIRPATH = "E:/college_project/dataset/" # change to new dataset `dirpath`
PREPROCESSING_RESULTS_DIRPATH = "E:/" # change accordingly


def histTransform(r, alpha, beta):
    n = -(r - alpha)
    d = beta
    s = 1 / (1 + np.exp(n/d))
    return s


def eme(X, patch_size):
    m, n = X.shape[:2]

    r = m // patch_size
    c = n // patch_size

    E = 0.
    B1 = np.zeros((patch_size, patch_size))
    m1 = 0
    for m in range(r):
        n1 = 0
        for n in range(c):
            B1 = X[m1: m1+patch_size, n1: n1+patch_size]
            b_min = np.min(B1)
            b_max = np.max(B1)
            if b_min > 0:
                b_ratio = b_max / b_min
                E = E + 20. * np.log(b_ratio)
            n1 = n1 + patch_size
        m1 = m1 + patch_size
    E = (E / r) / c
    return E


def preprocess(image, alpha=0.72, beta=0.30):
    # Normalize the image
    normalized_img = cv2.normalize(image, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Apply histogram based transformation (s-curve)
    transformed_img = histTransform(normalized_img, alpha, beta)
    
    # Get EME value of original image
    original_eme = eme(image, patch_size=7)
    # Get EME value of transformed image 
    new_eme = eme(transformed_img, patch_size=7)
        
    if original_eme < new_eme:
        return transformed_img
    else:
        return image    



for folder in os.listdir(DATASET_DIRPATH):
    for fname in os.listdir(os.path.join(DATASET_DIRPATH, folder)):
        
        CURRENT_PATH = os.path.join(DATASET_DIRPATH, folder, fname)
        DESTINATION_PATH = os.path.join(PREPROCESSING_RESULTS_DIRPATH, folder, fname)
        
        # Read image file
        image = Image.open(CURRENT_PATH)
        image = np.array(image)
        
        # Preprocess the image
        result = preprocess(image)
        
        # Save the resultant image
        imagefile = Image.fromarray(result)
        imagefile.save(DESTINATION_PATH)