#run the code using streamlit run filename.py in terminal

import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Demo image
demo = 'pic1.jpg'

# Body parts and their indices
bodyparts = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "Rknee": 9, 
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14, 
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

# Pairs for drawing the skeleton
posepair = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"],
    ["Neck", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"],
    ["Neck", "Nose"], ["Nose", "REye"], ["REye", "REar"],
    ["Nose", "LEye"], ["LEye", "LEar"]
]

# Input dimensions for the network
width = 360
height = 360
inwidth = width
inheight = height

# Load the pre-trained model
net = cv2.dnn.readNetFromTensorflow("out.pb")

# Streamlit App Title
st.title("Human Pose Estimation with OpenCV")
st.text('Make sure the image is clearly visible.')

# File uploader
imgbuffer = st.file_uploader("Upload an image with clear visuals", type=['jpg', 'jpeg', 'png'])

# Load image from file uploader or use the demo image
if imgbuffer is not None:
    img = np.array(Image.open(imgbuffer))
else:
    img = np.array(Image.open(demo))

# Display original image
st.subheader("Original Image")
st.image(img, caption=f"Original Image", use_column_width=True)

# Threshold slider
thres = st.slider('Threshold for detecting key points', min_value=0, value=20, max_value=100)
thres = thres / 100

# Pose detection function
@st.cache
def posedetect(frame, threshold):
    # Frame dimensions
    fw = frame.shape[1]
    fh = frame.shape[0]

    # Prepare the input for the network
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inwidth, inheight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # Take only the first 19 parts

    # Ensure that the output matches the bodyparts dictionary
    assert len(bodyparts) == out.shape[1]

    # Points list to store detected points
    points = []
    for i in range(len(bodyparts)):
        heatmap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatmap)
        x = (fw * point[0]) / out.shape[3]
        y = (fh * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > threshold else None)

    # Draw skeleton
    for pair in posepair:
        frm = pair[0]
        to = pair[1]
        
        if frm in bodyparts and to in bodyparts:  # Check if both points exist in bodyparts
            idfrom = bodyparts[frm]
            idto = bodyparts[to]

            if points[idfrom] and points[idto]:
                cv2.line(frame, points[idfrom], points[idto], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idfrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idto], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    
    t, _ = net.getPerfProfile()
    return frame

# Run pose detection
res = posedetect(img, thres)

# Display the results
st.subheader('Estimated Positions')
st.image(res, caption=f"Estimated Positions", use_column_width=True)
