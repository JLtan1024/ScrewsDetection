import asyncio
import sys

if sys.platform == "win32" and (3, 8, 0) <= sys.version_info < (3, 9, 0):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from collections import Counter
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Set title
st.title("🔍 Real-Time Screw Detection and Measurement (YOLOv11 OBB)")

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
IOU_THRESHOLD = st.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.7, step=0.05)

LABEL_FONT_SIZE = 20
BORDER_WIDTH = 3

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def get_text_size(draw, text, font):
    if hasattr(draw, 'textbbox'):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        return draw.textsize(text, font=font)

def non_max_suppression(detections, iou_threshold):
    """Improved NMS for OBB that keeps multiple non-overlapping boxes"""
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

# Initialize session state
if 'model' not in st.session_state:
    try:
        st.session_state.model = YOLO("yolo11-obb12classes.pt")
    except Exception as e:
        st.error(f"Error loading YOLO OBB model: {e}")
        st.stop()

# Initialize detection summary in session state
if 'detection_summary' not in st.session_state:
    st.session_state.detection_summary = Counter()

# Initialize font
try:
    font = ImageFont.truetype("arial.ttf", LABEL_FONT_SIZE)
except:
    try:
        font = ImageFont.truetype("LiberationSans-Regular.ttf", LABEL_FONT_SIZE)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", LABEL_FONT_SIZE)
        except:
            font = ImageFont.load_default()
            if hasattr(font, 'size'):
                font.size = LABEL_FONT_SIZE

def process_frame(frame):
    img = frame.to_ndarray(format="bgr24")
    pil_image = Image.fromarray(img)
    
    try:
        results = st.session_state.model(img)
        
        if results:
            result = results[0]
            filtered_detections = non_max_suppression(result.obb, IOU_THRESHOLD)
            
            draw = ImageDraw.Draw(pil_image)
            px_to_mm_ratio = None
            coin_detected = False
            detected_objects = []

            # Find coin for scaling
            for detection in filtered_detections:
                if len(detection.cls) > 0 and int(detection.cls[0]) == COIN_CLASS_ID and len(detection.xywhr) > 0:
                    coin_xywhr = detection.xywhr[0]
                    width_px = coin_xywhr[2]
                    height_px = coin_xywhr[3]
                    avg_px_diameter = (width_px + height_px) / 2
                    if avg_px_diameter > 0:
                        px_to_mm_ratio = COIN_DIAMETER_MM / avg_px_diameter
                        coin_detected = True
                    break

            for detection in filtered_detections:
                if len(detection.cls) > 0 and len(detection.xywhr) > 0 and len(detection.xyxy) > 0:
                    class_id = int(detection.cls[0])
                    confidence = detection.conf[0]
                    x1, y1, x2, y2 = map(int, detection.xyxy[0])
                    class_name = CLASS_NAMES.get(class_id, f"Class {int(class_id)}")
                    color = CATEGORY_COLORS.get(class_name, (0, 255, 0))

                    label_text = f"{class_name}"
                    if class_id != COIN_CLASS_ID:
                        detected_objects.append(class_name)

                    if class_id == COIN_CLASS_ID and coin_detected and px_to_mm_ratio:
                        diameter_px = (x2 - x1 + y2 - y1) / 2
                        diameter_mm = diameter_px * px_to_mm_ratio
                        label_text += f", Dia: {diameter_mm:.2f}mm"
                    elif class_id != COIN_CLASS_ID and coin_detected and px_to_mm_ratio:
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

                    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=BORDER_WIDTH)
                    text_width, text_height = get_text_size(draw, label_text, font)
                    draw.rectangle([(x1, y1 - text_height - 5), (x1 + text_width + 5, y1)], fill=color)
                    draw.text((x1 + 2, y1 - text_height - 3), label_text, fill=(255, 255, 255), font=font)

            # Update detection summary
            st.session_state.detection_summary.update(detected_objects)
            
    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
    
    return av.VideoFrame.from_ndarray(np.array(pil_image), format="bgr24")

# Main app
st.header("Real-Time Detection")
st.write("Point your camera at screws/nuts to detect and measure them. A 10sen coin is needed for scale.")

# WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=process_frame,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Display detection summary
st.subheader("✨ Real-Time Detection Summary ✨")
summary_placeholder = st.empty()

# Periodically update the summary
if webrtc_ctx.state.playing:
    while True:
        if st.session_state.detection_summary:
            summary_text = "Detected the following screws/nuts:\n"
            for name, count in st.session_state.detection_summary.most_common():
                color = '#%02x%02x%02x' % CATEGORY_COLORS.get(name, (0, 255, 0))
                summary_text += f"- <span style='color: {color}'>{name}:</span> **{count}**\n"
            summary_placeholder.markdown(summary_text, unsafe_allow_html=True)
        else:
            summary_placeholder.info("No screws or nuts detected yet.")
        
        # Small delay to prevent high CPU usage
        import time
        time.sleep(1)
