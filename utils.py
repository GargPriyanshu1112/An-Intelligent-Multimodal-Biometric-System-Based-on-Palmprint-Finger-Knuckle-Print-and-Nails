# Import dependencies
import os
from PIL import Image
import numpy as np
import re

from roi_extraction_utils import get_inner_hand_surface_ROI



def load_palmar_data(dirpath):
    rois, labels = [], []

    for fname in os.listdir(dirpath):
        # Read the content of the file
        image = Image.open(os.path.join(dirpath, fname))
        image = np.array(image)
        # Extract ROI (palmprint)
        roi = get_inner_hand_surface_ROI(image)
        # Assign image-id as its label
        label = re.search("[0-9]+", fname)
        label = int(label.group()) # label.dtype == 'int32'

        if roi is not None:
            rois.append(roi)
            labels.append(label)
    
    rois, labels = np.array(rois), np.array(labels)
    return rois, labels
