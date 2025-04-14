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

# Initialize session state
if 'model' not in st.session_state:
    try:
        st.session_state.model = YOLO("yolo11-obb.pt")
    except Exception as e:
        st.error(f"Error loading YOLO OBB model: {e}")
        st.stop()

# Image input method
input_method = st.radio(
    "Choose Image Input Method",
    ("Upload Image", "Use Camera"),
    index=0
)

# Image input based on selection
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
    # Convert image to numpy array
    processed_image = np.array(image)

    # Run YOLO OBB detection
    try:
        results = st.session_state.model(processed_image)

        if not results:
            st.warning("No detections found")
            st.stop()

        result = results[0]  # Get first (and only) set of results for the image

        # Display detection results with bounding boxes and labels
        if hasattr(result, 'plot'):
            plotted_img = result.plot()[:, :, ::-1]  # Convert BGR to RGB
            st.image(plotted_img, caption="YOLO v11 OBB Detection", use_container_width=True)
        else:
            st.error("Result object doesn't have plot method")
            st.stop()

        # Get OBB detections (list of OBB objects)
        obb_detections = result.obb
        st.write("Raw OBB Detections:")
        st.write(obb_detections)

        coin_detection = None
        other_detections = []

        # Separate coin and other detections
        for detection in obb_detections:
            if len(detection.cls) > 0 and int(detection.cls[0]) == COIN_CLASS_ID:
                coin_detection = detection
            elif len(detection.cls) > 0:
                other_detections.append(detection)

        if coin_detection:
            # Calculate pixel-to-mm ratio using the coin
            if len(coin_detection.xywhr) > 0:
                coin_xywhr = coin_detection.xywhr[0]
                width_px = coin_xywhr[2]
                height_px = coin_xywhr[3]
                avg_px_diameter = (width_px + height_px) / 2
                if avg_px_diameter > 0:
                    px_to_mm_ratio = COIN_DIAMETER_MM / avg_px_diameter
                    st.write(f"Pixel to mm ratio: {px_to_mm_ratio:.4f}")

                    # Measure other objects (screws/nuts)
                    st.subheader("📏 Screw/Nut Measurements:")
                    if other_detections:
                        for detection in other_detections:
                            if len(detection.cls) > 0 and len(detection.xywhr) > 0:
                                class_id = int(detection.cls[0])
                                xywhr = detection.xywhr[0]
                                width_px = xywhr[2]
                                height_px = xywhr[3]
                                length_px = max(width_px, height_px)
                                length_mm = length_px * px_to_mm_ratio
                                st.write(f"Class {class_id} object length: {length_mm:.2f} mm (approx.)")
                            else:
                                st.warning("Incomplete data for a detected object.")
                    else:
                        st.info("No screws/nuts detected.")
                else:
                    st.warning("Detected coin has zero diameter, cannot calculate ratio.")
            else:
                st.warning("Coin detection data is incomplete.")
        else:
            st.warning("No 10sen coin detected - cannot calculate measurements without reference.")

    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
