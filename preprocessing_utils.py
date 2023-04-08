# Import dependencies
import numpy as np
import cv2


def preprocess(image):
    # Convert the image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Apply CLAHE on the L channel of LAB color space
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8))
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)

    # Merge the CLAHE enhanced L channel with the original A and B channels
    lab_clahe = cv2.merge((l_clahe, a, b))

    # Convert the image back to RGB color space
    clahe_rgb = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    # Normalize the image
    result = clahe_rgb / 255.
    return result



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
