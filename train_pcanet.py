# Import dependenceis
import os 
import timeit
from PIL import Image
import numpy as np
import re
from sklearn.svm import SVC

import pcanet as net
from preprocessing_utils import preprocess
from roi_extraction_utils import get_inner_hand_surface_ROI
from utils import load_palmar_data


DATASET_DIRPATH = "/content/drive/MyDrive/dataset"
PALMAR_TRAIN_DIRPATH = os.path.join(DATASET_DIRPATH, "palmar", "train")
PALMAR_TEST_DIRPATH = os.path.join(DATASET_DIRPATH, "palmar", "test")
DORSAL_TRAIN_DIRPATH = os.path.join(DATASET_DIRPATH, "dorsal", "train")
DORSAL_TEST_DIRPATH = os.path.join(DATASET_DIRPATH, "dorsal", "test")


X_train, y_train = load_palmar_data(PALMAR_TRAIN_DIRPATH) 


def train(train_set):
    images_train, y_train = train_set

    print("Training PCANet")

    # Instanitate PCANet class
    pcanet = net.PCANet(
        image_shape=224,
        filter_shape_l1=4, step_shape_l1=1, n_l1_output=7,
        filter_shape_l2=4, step_shape_l2=1, n_l2_output=7,
        filter_shape_pooling=7, step_shape_pooling=1 # overlapping, can be change to 7 (non-overlapping)
    )

    pcanet.validate_structure()

    t1 = timeit.default_timer()
    pcanet.fit(images_train)
    t2 = timeit.default_timer()

    train_time = t2 - t1
    print(f"Train time: {train_time}")

    t1 = timeit.default_timer()
    X_train = pcanet.transform(images_train)
    t2 = timeit.default_timer()

    transform_time = t2 - t1
    print(f"Transform time: {transform_time}")
    
    print("Training the classifier")
    classifier = SVC(C=10) # Classification
    classifier.fit(X_train, y_train)
    return pcanet, classifier




X = X_train.astype('float32')

pcanet, classifier = train((X, y_train))