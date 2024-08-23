import cv2 as cv
import numpy as np
from tkinter import Tk, Button, Label, filedialog
from PIL import Image, ImageTk

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
thr = 0.1
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

# Draw keypoints on the original image
for pair in POSE_PAIRS:
    partFrom = pair[0]
    partTo = pair[1]
    idFrom = BODY_PARTS[partFrom]
    idTo = BODY_PARTS[partTo]

    if points[idFrom] and points[idTo]:
        cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
        cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

# Convert the frame with keypoints to PIL Image for Tkinter display
display_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
display_image_pil = Image.fromarray(display_image)

# Create a blank image for saving keypoints only
keypoints_image = np.ones((frameHeight, frameWidth, 3), np.uint8) * 255

# Draw keypoints on the blank image
for pair in POSE_PAIRS:
    partFrom = pair[0]
    partTo = pair[1]
    idFrom = BODY_PARTS[partFrom]
    idTo = BODY_PARTS[partTo]

    if points[idFrom] and points[idTo]:
        cv.line(keypoints_image, points[idFrom], points[idTo], (0, 255, 0), 3)
        cv.ellipse(keypoints_image, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        cv.ellipse(keypoints_image, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

# Convert keypoints_image to PIL Image for saving
keypoints_image_pil = Image.fromarray(keypoints_image)

# Function to save keypoints-only image
def save_keypoints_image():
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if save_path:
        keypoints_image_pil.save(save_path)

# Set up the Tkinter GUI
root = Tk()
root.title("Pose Estimation")

# Convert PIL Image to ImageTk for Tkinter display
tk_display_image = ImageTk.PhotoImage(display_image_pil)
label = Label(root, image=tk_display_image)
label.pack()

# Add a button to save the keypoints-only image
save_button = Button(root, text="Save Keypoints Image", command=save_keypoints_image)
save_button.pack()

root.mainloop()
