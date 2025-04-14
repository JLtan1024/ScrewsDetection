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
COIN_DIAMETER_MM = 18.80  # 10sen coin diameter in mm
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
IOU_THRESHOLD = 0.5  # Increased threshold for better duplicate removal
LABEL_FONT_SIZE = 30  # Significantly increased font size for better visibility
BORDER_WIDTH = 4     # Increased border width for bounding boxes

# Improved NMS function that works with OBB detections
def non_max_suppression(detections, iou_threshold):
    if len(detections) == 0:
        return []
    
    # Convert detections to xyxy format
    boxes = []
    scores = []
    classes = []
    keep_detections = []
    
    for det in detections:
        if len(det.xyxy) > 0:
            box = det.xyxy[0].cpu().numpy()
            boxes.append(box)
            scores.append(det.conf[0].cpu().numpy())
            classes.append(det.cls[0].cpu().numpy())
            keep_detections.append(det)
    
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    
    # Sort by confidence score (descending)
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]
    classes = classes[order]
    keep_detections = [keep_detections[i] for i in order]
    
    keep = []
    
    while boxes.shape[0] > 0:
        # Keep the box with the highest score
        keep.append(0)
        
        if boxes.shape[0] == 1:
            break
            
        # Calculate IoU between the first box and remaining boxes
        ious = []
        for i in range(1, boxes.shape[0]):
            iou = calculate_iou(boxes[0], boxes[i])
            ious.append(iou)
        
        ious = np.array(ious)
        
        # Remove boxes with IoU > threshold and same class
        same_class = (classes[1:] == classes[0])
        overlap = (ious > iou_threshold)
        remove = np.logical_and(same_class, overlap)
        
        # Keep boxes where remove is False
        keep_indices = np.where(~remove)[0] + 1  # +1 because we skipped index 0
        
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        classes = classes[keep_indices]
        keep_detections = [keep_detections[i] for i in keep_indices]
    
    return [keep_detections[i] for i in keep]

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

        # Apply our improved NMS
        filtered_detections = non_max_suppression(result.obb, IOU_THRESHOLD)

        # Prepare image for drawing with PIL
        pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a larger font
        try:
            # Try different fonts that might be available
            try:
                font = ImageFont.truetype("arial.ttf", LABEL_FONT_SIZE)
            except:
                try:
                    font = ImageFont.truetype("LiberationSans-Regular.ttf", LABEL_FONT_SIZE)
                except:
                    try:
                        font = ImageFont.truetype("DejaVuSans.ttf", LABEL_FONT_SIZE)
                    except:
                        # If no specific font is found, use default with increased size
                        font = ImageFont.load_default()
                        font.size = LABEL_FONT_SIZE
        except Exception as e:
            st.warning(f"Couldn't load preferred font: {e}")
            font = ImageFont.load_default()

        px_to_mm_ratio = None
        coin_detected = False
        detected_objects = []

        # Calculate pixel to mm ratio if coin is detected
        for detection in filtered_detections:
            if len(detection.cls) > 0 and int(detection.cls[0]) == COIN_CLASS_ID and len(detection.xywhr) > 0:
                coin_xywhr = detection.xywhr[0]
                width_px = coin_xywhr[2]
                height_px = coin_xywhr[3]
                avg_px_diameter = (width_px + height_px) / 2
                if avg_px_diameter > 0:
                    px_to_mm_ratio = COIN_DIAMETER_MM / avg_px_diameter
                    coin_detected = True
                break  # Assuming only one coin for reference

        for detection in filtered_detections:
            if len(detection.cls) > 0 and len(detection.xywhr) > 0 and len(detection.xyxy) > 0:
                class_id = int(detection.cls[0])
                confidence = detection.conf[0]
                x1, y1, x2, y2 = map(int, detection.xyxy[0])

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
                    width_px = xywhr[2]
                    height_px = xywhr[3]
                    length_px = max(width_px, height_px)
                    length_mm = length_px * px_to_mm_ratio
                    label_text += f", Length: {length_mm:.2f}mm"
                elif class_id != COIN_CLASS_ID:
                    label_text += ", Length: N/A (No Coin)"
                elif class_id == COIN_CLASS_ID:
                    label_text += ", Dia: N/A (No Ratio)"

                # Draw bounding box with thicker border
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=BORDER_WIDTH)
                
                # Draw text with background for better visibility
                text_width, text_height = draw.textsize(label_text, font=font)
                draw.rectangle([(x1, y1 - text_height - 5), (x1 + text_width + 5, y1)], fill=color)
                draw.text((x1 + 2, y1 - text_height - 3), label_text, fill=(255, 255, 255), font=font)

        st.image(pil_image, caption="Detected Objects with Info", use_column_width=True)

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
