import numpy as np

camera_matrix = np.load("camera_matrix.npy")
print(f"Camera\n", camera_matrix)
dist = np.load("dist.npy")
print(f"Distortion\n", dist)
error = np.load("error.npy")
print(f"error\n", error)
