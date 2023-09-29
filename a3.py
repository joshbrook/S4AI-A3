import numpy as np
import cv2

# Basic characters
S = np.array(
    [
        [0.99, 0.99, 0.99, 0.99, 0.99],
        [0.99, 0, 0, 0, 0],
        [0.99, 0.99, 0.99, 0.99, 0.99],
        [0, 0, 0, 0, 0.99],
        [0.99, 0.99, 0.99, 0.99, 0.99]
    ]
)

J = np.array(
    [
        [0.99, 0.99, 0.99, 0.99, 0.99],
        [0, 0, 0.99, 0, 0],
        [0, 0, 0.99, 0, 0],
        [0.99, 0, 0.99, 0, 0],
        [0.99, 0.99, 0.99, 0, 0]
    ]
)

H = np.array(
    [
        [0.99, 0, 0, 0, 0.99],
        [0.99, 0, 0, 0, 0.99],
        [0.99, 0.99, 0.99, 0.99, 0.99],
        [0.99, 0, 0, 0, 0.99],
        [0.99, 0, 0, 0, 0.99]
    ]
)

# Character variation 1 (emphasizing strokes)
S1 = np.array(
    [
        [0.99, 0.99, 0.99, 0.7, 0.5],
        [0.99, 0, 0, 0, 0],
        [0.99, 0.99, 0.99, 0.99, 0.99],
        [0, 0, 0, 0, 0.99],
        [0.5, 0.7, 0.5, 0.6, 0.99]
    ]
)

J1 = np.array(
    [
        [0.5, 0.9, 0.99, 0.8, 0.6],
        [0, 0, 0.99, 0, 0],
        [0, 0, 0.99, 0, 0],
        [0.4, 0, 0.99, 0, 0],
        [0.6, 0.8, 0.99, 0, 0]
    ]
)

H1 = np.array(
    [
        [0.6, 0, 0, 0, 0.99],
        [0.8, 0, 0, 0, 0.99],
        [0.8, 0.99, 0.99, 0.99, 0.99],
        [0.7, 0, 0, 0, 0.99],
        [0.6, 0, 0, 0, 0.99]
    ]
)

# Character variation 2 (blur)
def box_kernel(size):
    k = np.ones((size, size), np.float32)/(size**2)
    return k


S2 = cv2.filter2D(S1, -1, box_kernel(2)) # Should we blur more?

J2 = cv2.filter2D(J1, -1, box_kernel(2))

H2 = cv2.filter2D(H1, -1, box_kernel(2))

# Character variation 3 (noise)
noise = np.random.normal(0, 0.5, S.shape)
S3 = S1 + noise
S3 = np.clip(S3,0,1)

noise = np.random.normal(0, 0.5, J.shape)
J3 = J1 + noise
J3 = np.clip(J3,0,1)

noise = np.random.normal(0, 0.5, H.shape)
H3 = H1 + noise
H3 = np.clip(H3,0,1)

# Character variation 4 (blur + noise)
noise = np.random.normal(0, 0.5, S.shape)
S4 = S2 + noise
S4 = np.clip(S4,0,1)

noise = np.random.normal(0, 0.5, J.shape)
J4 = J2 + noise
J4 = np.clip(J4,0,1)

noise = np.random.normal(0, 0.5, H.shape)
H4 = H2 + noise
H4 = np.clip(H4,0,1)