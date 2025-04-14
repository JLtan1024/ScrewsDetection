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

# Load model with caching
@st.cache_resource
def load_model():
    try:
        return YOLO("yolo11-obb.pt")
    except Exception as e:
        st.error(f"Error loading YOLO OBB model: {e}")
        st.stop()

model = load_model()

# Image input method
input_method = st.radio(
    "Choose Image Input Method",
    ("Upload Image", "Use Camera"),
    index=0
)

# Process image
image = None
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
elif input_method == "Use Camera":
    camera_input = st.camera_input("Take a Picture")
    if camera_input is not None:
        image = Image.open(camera_input)

if image:
    # Convert to numpy array
    img_array = np.array(image)
    
    # Run detection
    try:
        results = model(img_array)
        
        if not results:
            st.warning("No objects detected")
            st.stop()
            
        # Get first result
        result = results[0]
        
        # Check if OBB results exist
        if not hasattr(result, 'obb'):
            st.warning("No oriented bounding box results found")
            st.stop()
            
        # Plot and display results
        plotted_img = result.plot()[:, :, ::-1]  # BGR to RGB
        st.image(plotted_img, caption="Detection Results", use_container_width=True)
        
        # Get OBB data
        obb = result.obb
        if not obb:
            st.warning("No OBB data available")
            st.stop()
            
        # Find coin for reference
        coin_detections = [d for d in obb if int(d.cls) == COIN_CLASS_ID]
        
        if not coin_detections:
            st.warning("No 10sen coin detected - measurements unavailable")
            st.stop()
            
        # Use first coin found
        coin = coin_detections[0]
        coin_size = max(coin.xywh[2], coin.xywh[3])  # Use largest dimension
        px_to_mm = COIN_DIAMETER_MM / coin_size
        
        # Measure other objects
        screw_measurements = []
        for det in obb:
            if int(det.cls) != COIN_CLASS_ID:
                size_px = max(det.xywh[2], det.xywh[3])
                size_mm = size_px * px_to_mm
                screw_measurements.append((int(det.cls), size_mm))
        
        # Display measurements
        if screw_measurements:
            st.subheader("📏 Measurement Results")
            for class_id, length_mm in screw_measurements:
                st.write(f"Class {class_id}: {length_mm:.2f} mm")
        else:
            st.warning("No screws/nuts detected (only coin found)")
            
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
