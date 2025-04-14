import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from collections import Counter

# Set title
st.title("🔍 Screw Detection and Measurement (YOLOv11 OBB)")

# Constants
COIN_CLASS_ID = 13  # 10sen coin
COIN_DIAMETER_MM = 20.60  # 10sen coin diameter in mm
CLASS_NAMES = {
    0: 'long lag screw',
    1: 'wood screw',
    2: 'lag wood screw',
    3: 'short wood screw',
    4: 'shiny screw',
    5: 'black oxide screw',
    6: 'nut',
    7: 'bolt',
    8: 'large nut',
    9: 'nut',
    10: 'nut',
    11: 'machine screw',
    12: 'short machine screw',
    13: '10sen Coin'
}
CATEGORY_COLORS = {
    'long lag screw': (255, 0, 0),     # Red
    'wood screw': (0, 255, 0),        # Green
    'lag wood screw': (0, 0, 255),     # Blue
    'short wood screw': (255, 255, 0),  # Yellow
    'shiny screw': (255, 0, 255),     # Magenta
    'black oxide screw': (0, 255, 255), # Cyan
    'nut': (128, 0, 128),            # Purple
    'bolt': (255, 165, 0),           # Orange
    'large nut': (128, 128, 0),       # Olive
    'machine screw': (0, 128, 128),    # Teal
    'short machine screw': (128, 0, 0), # Maroon
    '10sen Coin': (192, 192, 192)      # Silver
}

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

        result = results[0]

        # Prepare image for drawing with PIL
        pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        px_to_mm_ratio = None
        coin_detected = False
        detected_objects = []

        # Calculate pixel to mm ratio if coin is detected
        for detection in result.obb:
            if len(detection.cls) > 0 and int(detection.cls[0]) == COIN_CLASS_ID and len(detection.xywhr) > 0:
                coin_xywhr = detection.xywhr[0]
                width_px = coin_xywhr[2]
                height_px = coin_xywhr[3]
                avg_px_diameter = (width_px + height_px) / 2
                if avg_px_diameter > 0:
                    px_to_mm_ratio = COIN_DIAMETER_MM / avg_px_diameter
                    coin_detected = True
                break  # Assuming only one coin for reference

        for detection in result.obb:
            if len(detection.cls) > 0 and len(detection.xywhr) > 0 and len(detection.xyxy) > 0:
                class_id = int(detection.cls[0])
                confidence = detection.conf[0]
                xyxy = detection.xyxy[0]
                x1, y1, x2, y2 = map(int, xyxy)

                class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
                color = CATEGORY_COLORS.get(class_name, (0, 255, 0))  # Default to green

                label_text = f"{class_name}"
                detected_objects.append(class_name)

                if class_id == COIN_CLASS_ID and coin_detected and px_to_mm_ratio is not None:
                    diameter_px = (x2 - x1 + y2 - y1) / 2  # Approximate diameter from AABB
                    diameter_mm = diameter_px * px_to_mm_ratio
                    label_text += f", Dia: {diameter_mm:.2f}mm"
                elif class_id != COIN_CLASS_ID and coin_detected and px_to_mm_ratio is not None:
                    xywhr = detection.xywhr[0]
                    width_px = xywhr[2]
                    height_px = xywhr[3]
                    length_px = max(width_px, height_px)
                    length_mm = length_px * px_to_mm_ratio
                    label_text += f", Length: {length_mm:.2f}mm"
                elif class_id != COIN_CLASS_ID:
                    label_text += ", Length: N/A (No Coin)"
                elif class_id == COIN_CLASS_ID:
                    label_text += ", Dia: N/A (No Ratio)"

                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
                draw.text((x1, y1 - 10), label_text, fill=(255, 255, 255), font=font)

        st.image(pil_image, caption="Detected Objects with Info", use_container_width=True)

        # Summarize the number of each type of screw detected
        st.subheader("Detection Summary:")
        screw_counts = Counter(detected_objects)
        for name, count in screw_counts.items():
            if name != CLASS_NAMES.get(COIN_CLASS_ID):  # Don't count the coin
                st.write(f"- {name}: {count}")

    except Exception as e:
        st.error(f"Error during detection or processing: {e}")
