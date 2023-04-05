# Import dependencies
import os
from PIL import Image
import numpy as np
import re
import pickle

from preprocessing_utils import preprocess
from roi_extraction_utils import get_inner_hand_surface_ROI, get_landmark_ROI



def load_palmar_data(dirpath, roi_h, roi_w):
    rois, labels = [], []

    for fname in os.listdir(dirpath):
        # Read the content of the file
        image = Image.open(os.path.join(dirpath, fname))
        image = np.array(image)
        # Extract ROI (palmprint)
        roi = get_inner_hand_surface_ROI(image, roi_h, roi_w)
        # Assign image-id as its label
        label = re.search("[0-9]+", fname)
        label = int(label.group()) # label.dtype == 'int32'

        if roi is not None:
            roi = preprocess(roi)
            rois.append(roi)
            labels.append(label)
    
    rois, labels = np.array(rois), np.array(labels)
    return rois, labels



def load_dorsal_data(dirpath, landmark):
    rois, labels = [], []

    for fname in os.listdir(dirpath):
        # Read the content of the file
        image = Image.open(os.path.join(dirpath, fname))
        image = np.array(image)
        # Extract landmark ROI
        roi = get_landmark_ROI(image)
        # Assign image-id as its label
        label = re.search("[0-9]+", fname)
        label = int(label.group()) # label.dtype == 'int32'

        if roi is not None:
            roi = preprocess(roi, gray_scale=False)
            rois.append(roi)
            labels.append(label)
    
    rois, labels = np.array(rois), np.array(labels)
    return rois, labels



def save_model(model, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


        
def load_model(filepath):
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model
