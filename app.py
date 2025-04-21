import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import time
from ultralytics import YOLO

# Constants
COIN_CLASS_ID = 11
COIN_DIAMETER_MM = 18.80
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

# Initialize model
@st.cache_resource
def load_model():
    return YOLO("yolo11-obb12classes.pt")

model = load_model()

# Streamlit UI
st.title("🔍 Real-Time Screw Detection")
frame_placeholder = st.empty()
stop_button = st.button("Stop Webcam")

# Webcam settings
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480

# Webcam capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

px_to_mm_ratio = None
detection_history = []
fps = 0
prev_time = 0

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture frame")
        break
    
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # Convert color and process
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run detection
    results = model(rgb_frame)
    
    # Process detections
    processed_frame = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(processed_frame)
    
    current_detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Only process high-confidence detections
            if conf < 0.5:
                continue
                
            class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
            color = CATEGORY_COLORS.get(class_name, (0,255,0))
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=BORDER_WIDTH)
            
            # Calculate measurements if coin is detected
            if class_id == COIN_CLASS_ID:
                diameter_px = (x2 - x1 + y2 - y1) / 2
                px_to_mm_ratio = COIN_DIAMETER_MM / diameter_px
            
            # Add label
            label = f"{class_name} {conf:.2f}"
            if px_to_mm_ratio and class_id != COIN_CLASS_ID:
                length_px = max(x2-x1, y2-y1)
                length_mm = length_px * px_to_mm_ratio
                label += f" | {length_mm:.1f}mm"
            
            # Draw label with background for better visibility
            text_width, text_height = get_text_size(draw, label, font)
            draw.rectangle([x1, y1-text_height-5, x1+text_width+5, y1], fill=color)
            draw.text((x1+2, y1-text_height-3), label, fill=(255,255,255), font=font)
            
            current_detections.append(class_name)
    
    # Update detection history
    detection_history.extend(current_detections)
    if len(detection_history) > 100:  # Keep last 100 detections
        detection_history = detection_history[-100:]
    
    # Add FPS counter
    draw.text((10, 10), f"FPS: {fps:.1f}", fill=(0,255,0), font=font)
    
    # Display
    frame_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
    
    # Show summary
    if detection_history:
        counts = Counter(detection_history)
        summary_text = "### ✨ Detection Summary ✨\n"
        for name, count in counts.items():
            color = '#%02x%02x%02x' % CATEGORY_COLORS.get(name, (0, 255, 0))
            summary_text += f"- <span style='color: {color}'>{name}:</span> **{count}**\n"
        st.sidebar.markdown(summary_text, unsafe_allow_html=True)
    
    time.sleep(0.03)  # Control frame rate (~30fps)

cap.release()
st.success("Webcam stopped")
