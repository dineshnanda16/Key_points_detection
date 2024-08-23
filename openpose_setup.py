import cv2 as cv
import numpy as np
from tkinter import Tk, Button, Label, filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define constants
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

# Configuration
image_path = './dine1.png'  # Path to the input image
thr = 0.2
width = 368
height = 368

# Load pre-trained model
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

# Load image
frame = cv.imread(image_path)
if frame is None:
    print("Error loading image. Check the path.")
    exit()

frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

# Prepare the input for the network
net.setInput(cv.dnn.blobFromImage(frame, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
out = net.forward()
out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

assert(len(BODY_PARTS) == out.shape[1])

points = []
for i in range(len(BODY_PARTS)):
    heatMap = out[0, i, :, :]
    _, conf, _, point = cv.minMaxLoc(heatMap)
    x = (frameWidth * point[0]) / out.shape[3]
    y = (frameHeight * point[1]) / out.shape[2]
    points.append((int(x), int(y)) if conf > thr else None)

# Convert points to 3D for visualization (assuming a 2D plane for simplicity)
points_3d = np.zeros((len(points), 3))
for i, point in enumerate(points):
    if point is not None:
        points_3d[i] = [point[0], point[1], 0]  # z-coordinate is 0 in 2D case

# Plot the mesh
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot keypoints
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='r', marker='o')

# Draw connections
for pair in POSE_PAIRS:
    partFrom = pair[0]
    partTo = pair[1]
    idFrom = BODY_PARTS[partFrom]
    idTo = BODY_PARTS[partTo]

    if points[idFrom] and points[idTo]:
        x_values = [points_3d[idFrom][0], points_3d[idTo][0]]
        y_values = [points_3d[idFrom][1], points_3d[idTo][1]]
        z_values = [points_3d[idFrom][2], points_3d[idTo][2]]
        ax.plot(x_values, y_values, z_values, c='b')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
