# Import dependencies
import numpy as np
import cv2


def s_curve_transform(r, alpha, beta):
    n = -(r - alpha)
    d = beta
    s = 1 / (1 + np.exp(n/d))
    return s


def preprocess(image, alpha=0.82, beta=0.30, gray_scale=True):
    if gray_scale:
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    # Normalize the image
    normalized_img = cv2.normalize(image, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Apply Local s-Curve Transformation
    transformed_img = s_curve_transform(normalized_img, alpha, beta)
    return transformed_img


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

