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

        result = results[0]  # Get first (and only) result

        # Display detection results
        if hasattr(result, 'plot'):
            plotted_img = result.plot()[:, :, ::-1]  # Convert BGR to RGB
            st.image(plotted_img, caption="YOLO v11 OBB Detection", use_container_width=True)
        else:
            st.error("Result object doesn't have plot method")
            st.stop()

        # Get OBB detections
        if hasattr(result, 'obb'):
            obb_detections = result.obb.cpu().numpy() # Convert to numpy array for easier indexing
            st.write("Raw OBB Detections:")
            st.write(obb_detections.shape) # Inspect the structure of obb_detections
            st.write(obb_detections)
            

            # Calculate pixel-to-mm ratio using 10sen coin (class 13)
            coin_detections = [det for det in obb_detections if len(det) > 5 and int(det[5]) == COIN_CLASS_ID]

            # Calculate pixel-to-mm ratio using 10sen coin (class 13)
            coin_detections = [det for det in obb_detections if len(det) == 7 and int(det[6]) == COIN_CLASS_ID]

            if coin_detections:
                coin = coin_detections[0]
                # Width and height are at indices 2 and 3
                if len(coin) >= 4:
                    width_px = coin[2]
                    height_px = coin[3]
                    avg_px_diameter = (width_px + height_px) / 2
                    if avg_px_diameter > 0:
                        px_to_mm_ratio = COIN_DIAMETER_MM / avg_px_diameter
                    else:
                        st.warning("Detected coin has zero diameter, cannot calculate ratio.")
                        px_to_mm_ratio = None
                else:
                    st.warning("Coin detection data is incomplete.")
                    px_to_mm_ratio = None

                if px_to_mm_ratio is not None:
                    screw_lengths = []
                    for det in obb_detections:
                        if len(det) == 7:
                            class_id = int(det[6])
                            # Width and height for all objects are at indices 2 and 3
                            if class_id != COIN_CLASS_ID:
                                width_px = det[2]
                                height_px = det[3]
                                length_px = max(width_px, height_px)
                                length_mm = length_px * px_to_mm_ratio
                                screw_lengths.append((class_id, length_mm))
                            elif class_id != COIN_CLASS_ID:
                                st.warning(f"Incomplete data for non-coin detection.")

                    # Display measurements
                    st.subheader("📏 Screw Measurements:")
                    if screw_lengths:
                        for class_id, length_mm in screw_lengths:
                            st.write(f"Class {class_id} screw/nut length: {length_mm:.2f} mm")
                    else:
                        st.warning("No screws/nuts detected (only coin found)")
            else:
                st.warning("No 10sen coin detected - cannot calculate measurements without reference")
        else:
            st.error("No OBB detections found in results")
            st.stop()

    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
