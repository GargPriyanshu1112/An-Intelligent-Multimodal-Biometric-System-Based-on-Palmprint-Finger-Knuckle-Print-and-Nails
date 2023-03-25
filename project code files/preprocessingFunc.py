# Import dependencies
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import os


# Set up path
dataset_dirpath = "E:/college_project/dataset"


def get_random_image(dirpath):
    fname = random.choice(os.listdir(dirpath))
    fpath = os.path.join(dirpath, fname)
    file = plt.imread(fpath)
    return file


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
    normalized_img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Apply histogram based transformation (s-curve)
    transformed_img = histTransform(normalized_img, alpha, beta)
    
    # Get EME value of original image
    original_eme = eme(image, patch_size=7)
    # Get EME value of transformed image 
    new_eme = eme(transformed_img, patch_size=7)
        
    if original_eme < new_eme:
        return transformed_img, original_eme, new_eme
    else:
        return image, original_eme, new_eme
    









img = get_random_image(dataset_dirpath)
image1, oeme, neme = preprocess(img)
print(oeme, neme)

##
normalized_img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
transformed_img = histTransform(normalized_img, alpha=0.72, beta=0.30)
print(f"eme original: {eme(img, 7)}")
print(f"eme new_img : {eme(transformed_img, 7)}")
##


plt.subplot(1, 3, 1)
plt.imshow(img)
plt.axis(False)
plt.subplot(1, 3, 2)
plt.imshow(image1)
plt.axis(False)
plt.subplot(1, 3, 3)
plt.imshow(transformed_img)
plt.axis(False)

# print(f"eme original: {eme(img, 7)}")
# print(f"eme new_img : {eme(transformed_img, 7)}")



# new_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
