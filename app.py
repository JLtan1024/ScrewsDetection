import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import time
import tempfile
import sys
import asyncio
import threading

# Fix for Torch/Streamlit Compatibility in Python 3.12
if sys.version_info >= (3, 12):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if threading.current_thread() is threading.main_thread():
            asyncio.set_event_loop(asyncio.new_event_loop())

# Constants
COIN_CLASS_ID = 11  # 10sen coin
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
    9: 'machine screw',
    10: 'short machine screw',
    11: '10sen Coin'
}
CATEGORY_COLORS = {
    'long lag screw': (255, 0, 0),
    'wood screw': (0, 255, 0),
    'lag wood screw': (0, 0, 255),
    'short wood screw': (255, 255, 0),
    'shiny screw': (255, 0, 255),
    'black oxide screw': (0, 255, 255),
    'nut': (128, 0, 128),
    'bolt': (255, 165, 0),
    'large nut': (128, 128, 0),
    'machine screw': (0, 128, 128),
    'short machine screw': (128, 0, 0),
    '10sen Coin': (192, 192, 192)
}
LABEL_FONT_SIZE = 20
BORDER_WIDTH = 3

# Initialize model with error handling
@st.cache_resource
def load_model():
    try:
        model = YOLO("yolo11-obb12classes.pt")
        # Warm-up inference
        dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        _ = model(dummy_input)
        return model
    except Exception as e:
        st.error(f"Model initialization failed: {e}")
        st.stop()

try:
    from ultralytics import YOLO
    model = load_model()
except ImportError as e:
    st.error(f"Failed to import YOLO: {e}")
    st.stop()

# Streamlit UI
st.title("🔍 Screw Detection and Measurement (YOLOv11 OBB)")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    input_method = st.radio(
        "Input Source",
        ("Webcam", "Upload Image", "Upload Video"),
        index=0
    )
    
    st.subheader("Detection Parameters")
    IOU_THRESHOLD = st.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.7, step=0.05)
    CONFIDENCE_THRESHOLD = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, step=0.05)
    
    if input_method == "Webcam":
        WEBCAM_WIDTH = st.slider("Webcam Width", 320, 1920, 640, step=160)
        WEBCAM_HEIGHT = st.slider("Webcam Height", 240, 1080, 480, step=120)
        SHOW_FPS = st.checkbox("Show FPS", value=True)
    
    st.subheader("Display Options")
    SHOW_DETECTIONS = st.checkbox("Show Detections", value=True)
    SHOW_SUMMARY = st.checkbox("Show Summary", value=True)

def get_text_size(draw, text, font):
    if hasattr(draw, 'textbbox'):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    return draw.textsize(text, font=font)

def non_max_suppression(detections, iou_threshold):
    if len(detections) == 0:
        return []

    boxes = []
    scores = []
    classes = []

    for det in detections:
        if len(det.xyxy) > 0:
            boxes.append(det.xyxy[0].cpu().numpy())
            scores.append(det.conf[0].cpu().numpy())
            classes.append(det.cls[0].cpu().numpy())

    if not boxes:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    indices = np.argsort(scores)[::-1]
    keep_indices = []

    while len(indices) > 0:
        current = indices[0]
        keep_indices.append(current)
        rest = indices[1:]

        ious = []
        for i in rest:
            box1 = boxes[current]
            box2 = boxes[i]
            xA = max(box1[0], box2[0])
            yA = max(box1[1], box2[1])
            xB = min(box1[2], box2[2])
            yB = min(box1[3], box2[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            unionArea = box1Area + box2Area - interArea
            iou = interArea / unionArea if unionArea > 0 else 0.0
            ious.append(iou)

        ious = np.array(ious)
        same_class = (classes[rest] == classes[current])
        to_keep = ~(same_class & (ious > iou_threshold))
        indices = rest[to_keep]

    return [detections[i] for i in keep_indices]

def process_frame(frame, model, px_to_mm_ratio=None):
    results = model(frame, conf=CONFIDENCE_THRESHOLD)
    
    if not results:
        return frame, [], px_to_mm_ratio
    
    result = results[0]
    filtered_detections = non_max_suppression(result.obb, IOU_THRESHOLD)
    
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", LABEL_FONT_SIZE)
    except:
        font = ImageFont.load_default()
        if hasattr(font, 'size'):
            font.size = LABEL_FONT_SIZE

    detected_objects = []
    current_px_to_mm_ratio = px_to_mm_ratio
    
    # Find coin for scaling
    if current_px_to_mm_ratio is None:
        for detection in filtered_detections:
            if len(detection.cls) > 0 and int(detection.cls[0]) == COIN_CLASS_ID and len(detection.xywhr) > 0:
                coin_xywhr = detection.xywhr[0]
                width_px = coin_xywhr[2]
                height_px = coin_xywhr[3]
                avg_px_diameter = (width_px + height_px) / 2
                if avg_px_diameter > 0:
                    current_px_to_mm_ratio = COIN_DIAMETER_MM / avg_px_diameter
                break

    # Draw detections
    for detection in filtered_detections:
        if len(detection.cls) > 0 and len(detection.xywhr) > 0 and len(detection.xyxy) > 0:
            class_id = int(detection.cls[0])
            confidence = detection.conf[0]
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            class_name = CLASS_NAMES.get(class_id, f"Class {int(class_id)}")
            color = CATEGORY_COLORS.get(class_name, (0, 255, 0))

            label_text = f"{class_name} {confidence:.2f}"
            if class_id != COIN_CLASS_ID:
                detected_objects.append(class_name)

            if class_id == COIN_CLASS_ID and current_px_to_mm_ratio:
                diameter_px = (x2 - x1 + y2 - y1) / 2
                diameter_mm = diameter_px * current_px_to_mm_ratio
                label_text += f", Dia: {diameter_mm:.2f}mm"
            elif class_id != COIN_CLASS_ID and current_px_to_mm_ratio:
                xywhr = detection.xywhr[0]
                width_px = xywhr[2]
                height_px = xywhr[3]
                length_px = max(width_px, height_px)
                length_mm = length_px * current_px_to_mm_ratio
                label_text += f", Length: {length_mm:.2f}mm"

            if SHOW_DETECTIONS:
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=BORDER_WIDTH)
                text_width, text_height = get_text_size(draw, label_text, font)
                draw.rectangle([(x1, y1 - text_height - 5), (x1 + text_width + 5, y1)], fill=color)
                draw.text((x1 + 2, y1 - text_height - 3), label_text, fill=(255, 255, 255), font=font)

    return np.array(pil_image), detected_objects, current_px_to_mm_ratio

# Main processing
frame_placeholder = st.empty()
summary_placeholder = st.empty()

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        processed_frame, detected_objects, _ = process_frame(frame, model)
        frame_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
        
        if SHOW_SUMMARY and detected_objects:
            screw_counts = Counter(detected_objects)
            summary_text = "### ✨ Detection Summary ✨\n"
            for name, count in screw_counts.items():
                color = '#%02x%02x%02x' % CATEGORY_COLORS.get(name, (0, 255, 0))
                summary_text += f"- <span style='color: {color}'>{name}:</span> **{count}**\n"
            summary_placeholder.markdown(summary_text, unsafe_allow_html=True)

elif input_method == "Upload Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        px_to_mm_ratio = None
        all_detected_objects = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame, detected_objects, px_to_mm_ratio = process_frame(frame, model, px_to_mm_ratio)
            if detected_objects:
                all_detected_objects.extend(detected_objects)
            
            frame_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
            
            if SHOW_SUMMARY and all_detected_objects:
                screw_counts = Counter(all_detected_objects)
                summary_text = "### ✨ Detection Summary ✨\n"
                for name, count in screw_counts.items():
                    color = '#%02x%02x%02x' % CATEGORY_COLORS.get(name, (0, 255, 0))
                    summary_text += f"- <span style='color: {color}'>{name}:</span> **{count}**\n"
                summary_placeholder.markdown(summary_text, unsafe_allow_html=True)
            
            time.sleep(0.03)
            
        cap.release()

elif input_method == "Webcam":
    if 'cap' not in st.session_state or st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    
    stop_button = st.button("Stop Webcam")
    
    px_to_mm_ratio = None
    all_detected_objects = []
    fps = 0
    prev_time = 0
    
    while st.session_state.cap.isOpened() and not stop_button:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break
            
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        processed_frame, detected_objects, px_to_mm_ratio = process_frame(frame, model, px_to_mm_ratio)
        if detected_objects:
            all_detected_objects.extend(detected_objects)
        
        if SHOW_FPS:
            cv2.putText(
                processed_frame, 
                f"FPS: {fps:.1f}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
        
        frame_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
        
        if SHOW_SUMMARY and all_detected_objects:
            screw_counts = Counter(all_detected_objects)
            summary_text = "### ✨ Detection Summary ✨\n"
            for name, count in screw_counts.items():
                color = '#%02x%02x%02x' % CATEGORY_COLORS.get(name, (0, 255, 0))
                summary_text += f"- <span style='color: {color}'>{name}:</span> **{count}**\n"
            summary_placeholder.markdown(summary_text, unsafe_allow_html=True)
        
        time.sleep(0.033)  # ~30fps
    
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.success("Webcam stopped")
    
