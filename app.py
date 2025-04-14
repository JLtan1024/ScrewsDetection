import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Set title
st.title("🔍 Screw Detection and Measurement (YOLOv11 OBB)")

# Constants
COIN_CLASS_ID = 13  # 10sen coin
COIN_DIAMETER_MM = 20.60  # 10sen coin diameter in mm

# Load YOLO OBB model
yolo_obb_model_path = "yolo11-obb.pt"
try:
    yolo_obb_model = YOLO(yolo_obb_model_path)
except Exception as e:
    st.error(f"Error loading YOLO OBB model: {e}")
    st.stop()

# Image input method
option = st.radio("Choose Image Input Method", ("Upload an Image", "Take a Photo"))
image = None

if option == "Upload an Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
elif option == "Take a Photo":
    camera_input = st.camera_input("Take a Picture")
    if camera_input is not None:
        image = Image.open(camera_input)

if image:
    # Convert image to numpy array
    processed_image = np.array(image)
    
    # Run YOLO OBB detection
    yolo_obb_results = yolo_obb_model(processed_image)
    
    # Display detection results
    yolo_obb_image = Image.fromarray(yolo_obb_results[0].plot()[:, :, ::-1])
    st.image(yolo_obb_image, caption="YOLO v11 OBB Detection", use_column_width=True)

    # Calculate pixel-to-mm ratio using 10sen coin (class 13)
    obb_detections = yolo_obb_results[0].obb
    coin_detections = [det for det in obb_detections if int(det['cls']) == COIN_CLASS_ID]
    
    if coin_detections:
        # Use the first detected coin as reference
        coin = coin_detections[0]
        width_px = coin['xywh'][2]
        height_px = coin['xywh'][3]
        avg_px_diameter = (width_px + height_px) / 2
        px_to_mm_ratio = COIN_DIAMETER_MM / avg_px_diameter
        
        # Measure screw lengths
        screw_lengths = []
        for det in obb_detections:
            class_id = int(det['cls'])
            if class_id != COIN_CLASS_ID:  # Skip the coin
                width_px = det['xywh'][2]
                height_px = det['xywh'][3]
                length_px = max(width_px, height_px)
                length_mm = length_px * px_to_mm_ratio
                screw_lengths.append((class_id, length_mm))
        
        # Display measurements
        st.subheader("📏 Screw Measurements:")
        if screw_lengths:
            for class_id, length_mm in screw_lengths:
                st.write(f"Class {class_id} screw/nut length: {length_mm:.2f} mm")
        else:
            st.warning("No screws/nuts detected (only coin found)")
    else:
        st.warning("No 10sen coin detected - cannot calculate measurements without reference")
