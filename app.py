# to run this script: streamlit run app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import time
import tempfile
from ultralytics import YOLO
from streamlit_webrtc import RTCConfiguration, webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import supervision as sv

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

model = YOLO("yolo11-obb12classes.pt")


class VideoTransformer(VideoProcessorBase):
    def __init__(self):
        self.flip = False  # Example parameter to control frame processing

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the frame to a numpy array
        img = frame.to_ndarray(format="bgr24")

        # Example processing: Flip the frame horizontally if self.flip is True
        if self.flip:
            img = cv2.flip(img, 1)

        # Example processing: Add a rectangle overlay
        height, width, _ = img.shape
        cv2.rectangle(img, (50, 50), (width - 50, height - 50), (0, 255, 0), 2)

        # Convert the processed frame back to an av.VideoFrame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    input_method = st.radio(
        "Input Source",
        ("Webcam (Live Camera)", "Upload Image", "Upload Video"),
        index=0
    )
    IOU_THRESHOLD = st.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.7, step=0.05)
    CONFIDENCE_THRESHOLD = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, step=0.05)
    
    if input_method == "Webcam (Live Camera)":
        WEBCAM_WIDTH = st.slider("Webcam Width", 320, 1920, 640, step=160)
        WEBCAM_HEIGHT = st.slider("Webcam Height", 240, 1080, 480, step=120)
        SHOW_FPS = st.checkbox("Show FPS", value=True)
    
    SHOW_DETECTIONS = st.checkbox("Show Detections", value=True)
    SHOW_SUMMARY = st.checkbox("Show Summary", value=True)

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

def process_frame(frame, model, px_to_mm_ratio=None):
    """Process a single frame and return annotated image and detection data"""
    results = model(frame, conf=CONFIDENCE_THRESHOLD)
    st.write(f"Results: {results}")
    if not results:
        return frame, [], px_to_mm_ratio
    
    result = results[0]
    print("Object detected:", len(result))
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
        st.write(f"Detection: {detection}")
        if len(detection.cls) > 0 and len(detection.xywhr) > 0 and len(detection.xyxy) > 0:
            class_id = int(detection.cls[0])
            confidence = detection.conf[0]
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            class_name = CLASS_NAMES.get(class_id, f"Class {int(class_id)}")
            color = CATEGORY_COLORS.get(class_name, (0, 255, 0))

            label_text = f"{class_name}"
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
            elif class_id != COIN_CLASS_ID:
                label_text += ", Length: N/A (No Coin)"
            elif class_id == COIN_CLASS_ID:
                label_text += ", Dia: N/A (No Ratio)"

            if SHOW_DETECTIONS:
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=BORDER_WIDTH)
                text_width, text_height = get_text_size(draw, label_text, font)
                draw.rectangle([(x1, y1 - text_height - 5), (x1 + text_width + 5, y1)], fill=color)
                draw.text((x1 + 2, y1 - text_height - 3), label_text, fill=(255, 255, 255), font=font)

    return np.array(pil_image), detected_objects, current_px_to_mm_ratio

def get_webcam_frame():
    """Get frame from webcam with fallback to Streamlit camera"""
    # Try direct OpenCV capture first
    try:
        cap = cv2.VideoCapture(0)  # Open the default webcam
        if not cap.isOpened():
            st.warning("Webcam could not be opened. Please check your camera settings.")
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read frame from webcam.")
            cap.release()
            return None

        cap.release()
        return frame
    except Exception as e:
        st.warning(f"OpenCV webcam access failed: {e}")
        return None

    # Fallback to Streamlit's camera input
    try:
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            return cv2.imdecode(np.frombuffer(
                img_file_buffer.getvalue(), 
                np.uint8
            ), cv2.IMREAD_COLOR)
    except Exception as e:
        st.error(f"Camera capture failed: {e}")
    
    return None

# Main app
st.title("🔍 Screw Detection and Measurement (YOLOv11 OBB)")

frame_placeholder = st.empty()
summary_placeholder = st.empty()

if input_method == "Upload Image":
    st.subheader("Upload or Capture an Image")
    
    # Option to upload an image
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    # Option to capture an image using the camera
    captured_image = st.camera_input("Take a Picture")
    
    if uploaded_file is not None:
        # Process uploaded image
        image = Image.open(uploaded_file)
        frame = np.array(image)
    elif captured_image is not None:
        # Process captured image
        frame = cv2.imdecode(np.frombuffer(captured_image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    else:
        frame = None

    if frame is not None:
        processed_frame, detected_objects, _ = process_frame(frame, st.session_state.model)
        frame_placeholder.image(processed_frame, channels="RGB", use_container_width=True)
        
        if SHOW_SUMMARY and detected_objects:
            screw_counts = Counter(detected_objects)
            summary_text = "### ✨ Detection Summary ✨\n"
            for name, count in screw_counts.items():
                color = '#%02x%02x%02x' % CATEGORY_COLORS.get(name, (0, 255, 0))
                summary_text += f"- <span style='color: {color}'>{name}:</span> **{count}**\n"
            summary_placeholder.markdown(summary_text, unsafe_allow_html=True)
        elif SHOW_SUMMARY:
            summary_placeholder.info("No screws or nuts detected.")

elif input_method == "Upload Video":
    st.subheader("Upload or Capture a Video")
    
    # Option to upload a video
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    
    # Option to capture a video using the camera
    captured_video = st.camera_input("Record a Video")
    
    if uploaded_video is not None:
        # Process uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        video_path = tfile.name
    elif captured_video is not None:
        # Process captured video
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(captured_video.getvalue())
        video_path = tfile.name
    else:
        video_path = None

    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        px_to_mm_ratio = None
        all_detected_objects = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame, detected_objects, px_to_mm_ratio = process_frame(
                frame, st.session_state.model, px_to_mm_ratio
            )
            
            if detected_objects:
                all_detected_objects.extend(detected_objects)
            
            frame_placeholder.image(processed_frame, channels="RGB", use_container_width=True)
            
            if SHOW_SUMMARY and all_detected_objects:
                screw_counts = Counter(all_detected_objects)
                summary_text = "### ✨ Detection Summary ✨\n"
                for name, count in screw_counts.items():
                    color = '#%02x%02x%02x' % CATEGORY_COLORS.get(name, (0, 255, 0))
                    summary_text += f"- <span style='color: {color}'>{name}:</span> **{count}**\n"
                summary_placeholder.markdown(summary_text, unsafe_allow_html=True)
            elif SHOW_SUMMARY:
                summary_placeholder.info("No screws or nuts detected yet.")
            
            time.sleep(0.03)  # Control playback speed
            
        cap.release()

elif input_method == "Webcam (Live Camera)":
    st.subheader("Live Camera Detection")

    
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    video_processor  = VideoTransformer()
    # Start the webcam stream using streamlit-webrtc
    webrtc_streamer(
        key="example-video-callback",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback= video_processor.recv,
        media_stream_constraints={"video": True, "audio": False},
    )
