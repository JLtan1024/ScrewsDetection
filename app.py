import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from collections import Counter
from scipy.spatial.distance import cdist

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
IOU_THRESHOLD = 0.7  # Threshold for considering boxes as the same object
MIN_FONT_SIZE = 10
MAX_FONT_SIZE = 30
FONT_SCALING_FACTOR = 0.05

# Function to calculate Intersection over Union (IoU) for axis-aligned boxes
def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0

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

        px_to_mm_ratio = None
        coin_detected = False
        detected_objects = []
        processed_detections = []

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

        # Non-maximum suppression (NMS) by class
        if result.boxes is not None and len(result.boxes) > 0:
            unique_classes = np.unique(result.boxes.cls.cpu().numpy())
            for cls in unique_classes:
                class_indices = (result.boxes.cls == cls).cpu().numpy()
                class_boxes = result.boxes.xyxy.cpu().numpy()[class_indices]
                class_conf = result.boxes.conf.cpu().numpy()[class_indices]
                keep_indices = cv2.dnn.NMSBoxes(class_boxes.tolist(), class_conf.tolist(), 0.3, IOU_THRESHOLD)
                if len(keep_indices) > 0:
                    for i in keep_indices.flatten():
                        detection = result.obb[class_indices[i]]
                        box_xyxy = class_boxes[i]
                        processed_detections.append((detection, box_xyxy))
        elif result.obb is not None:
            processed_detections = [(det, det.xyxy[0].cpu().numpy()) for det in result.obb]


        for detection, box_xyxy in processed_detections:
            if len(detection.cls) > 0 and len(detection.xywhr) > 0 and len(detection.xyxy) > 0:
                class_id = int(detection.cls[0])
                confidence = detection.conf[0]
                x1, y1, x2, y2 = map(int, box_xyxy)
                width_px = x2 - x1
                height_px = y2 - y1
                object_area = width_px * height_px

                # Dynamic font size calculation
                dynamic_font_size = max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, int(object_area * FONT_SCALING_FACTOR)))

                try:
                    font = ImageFont.truetype("arial.ttf", dynamic_font_size)
                except IOError:
                    font = ImageFont.load_default()

                class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
                color = CATEGORY_COLORS.get(class_name, (0, 255, 0))  # Default to green

                label_text = f"{class_name}"
                if class_name != CLASS_NAMES.get(COIN_CLASS_ID):
                    detected_objects.append(class_name)

                if class_id == COIN_CLASS_ID and coin_detected and px_to_mm_ratio is not None:
                    diameter_px = (x2 - x1 + y2 - y1) / 2  # Approximate diameter from AABB
                    diameter_mm = diameter_px * px_to_mm_ratio
                    label_text += f", Dia: {diameter_mm:.2f}mm"
                elif class_id != COIN_CLASS_ID and coin_detected and px_to_mm_ratio is not None:
                    xywhr = detection.xywhr[0]
                    width_px_obb = xywhr[2]
                    height_px_obb = xywhr[3]
                    length_px = max(width_px_obb, height_px_obb)
                    length_mm = length_px * px_to_mm_ratio
                    label_text += f", Length: {length_mm:.2f}mm"
                elif class_id != COIN_CLASS_ID:
                    label_text += ", Length: N/A (No Coin)"
                elif class_id == COIN_CLASS_ID:
                    label_text += ", Dia: N/A (No Ratio)"

                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
                draw.text((x1, y1 - dynamic_font_size - 5), label_text, fill=(255, 255, 255), font=font)

        st.image(pil_image, caption="Detected Objects with Info", use_container_width=True)

        # Beautiful and organized summary
        st.subheader("✨ Detection Summary ✨")
        screw_counts = Counter(detected_objects)
        if screw_counts:
            st.markdown("Detected the following screws/nuts:")
            for name, count in screw_counts.items():
                st.markdown(f"- <span style='color: {CATEGORY_COLORS.get(name, 'green')}'>{name}:</span> **{count}**", unsafe_allow_html=True)
        else:
            st.info("No screws or nuts detected.")

    except Exception as e:
        st.error(f"Error during detection or processing: {e}")
