# to run this script: streamlit run app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import time
import tempfile
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import supervision as sv
import hashlib
import warnings
import math
from streamlit.components.v1 import html

# Suppress torch warnings
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

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

# Load YOLO model
model = YOLO("yolo11-obb12classes.pt")

# Initialize session state for tracking
if 'tracked_objects' not in st.session_state:
    st.session_state.tracked_objects = {}  # Changed from set to dict to store class info

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
    SHOW_ORIENTATION = st.checkbox("Show Orientation", value=True)
    RESET_COUNTER = st.button("Reset Detection Counter")

# Reset counter if button pressed
if RESET_COUNTER:
    st.session_state.tracked_objects = {}

def get_text_size(draw, text, font):
    if hasattr(draw, 'textbbox'):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        return draw.textsize(text, font=font)

def xywhr_to_corners(xywhr):
    """Convert xywhr format to four corner points of the rotated rectangle"""
    x, y, w, h, r = xywhr
    cos_r = math.cos(r)
    sin_r = math.sin(r)
    
    # Calculate half width and height
    half_w = w / 2
    half_h = h / 2
    
    # Calculate the four corners relative to center
    corners = np.array([
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h]
    ])
    
    # Rotate the corners
    rotation_matrix = np.array([
        [cos_r, -sin_r],
        [sin_r, cos_r]
    ])
    
    rotated_corners = np.dot(corners, rotation_matrix.T)
    
    # Translate corners to absolute position
    rotated_corners[:, 0] += x
    rotated_corners[:, 1] += y
    
    return rotated_corners.astype(int)

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
            box2Area = (box2[2] - box2[0]) * (box2[3] - box1[1])
            unionArea = box1Area + box2Area - interArea
            iou = interArea / unionArea if unionArea > 0 else 0.0
            ious.append(iou)

        ious = np.array(ious)
        same_class = (classes[rest] == classes[current])
        to_keep = ~(same_class & (ious > iou_threshold))
        indices = rest[to_keep]

    return [detections[i] for i in keep_indices]

def generate_object_id(bbox, class_name):
    """Generate a unique ID based on bbox and class"""
    x1, y1, x2, y2 = map(int, bbox)
    return hashlib.md5(f"{x1}{y1}{x2}{y2}{class_name}".encode()).hexdigest()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    try:
        # Initialize FPS counter in session state
        if 'fps_counter' not in st.session_state:
            st.session_state.fps_counter = {
                'last_time': time.time(),
                'frame_count': 0,
                'current_fps': 0
            }
        
        # Update FPS counter
        counter = st.session_state.fps_counter
        counter['frame_count'] += 1
        elapsed = time.time() - counter['last_time']
        
        # Update FPS every second
        if elapsed >= 1.0:
            counter['current_fps'] = counter['frame_count'] / elapsed
            counter['frame_count'] = 0
            counter['last_time'] = time.time()
        
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Show FPS if enabled
        if SHOW_FPS:
            fps_text = f"FPS: {counter['current_fps']:.1f}"
            cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        # Run YOLO OBB inference
        results = model(img, conf=CONFIDENCE_THRESHOLD)
        
        if results and len(results[0].obb) > 0:
            result = results[0]
            new_objects_detected = False
            # Find coin for scaling
            highest_confidence = 0
            for detection in result.obb:
                if len(detection.cls) > 0 and int(detection.cls[0]) == COIN_CLASS_ID and len(detection.xywhr) > 0:
                    confidence = detection.conf[0]
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        coin_xywhr = detection.xywhr[0].cpu().numpy()
                        width_px = coin_xywhr[2]
                        height_px = coin_xywhr[3]
                        avg_px_diameter = (width_px + height_px) / 2
                        if avg_px_diameter > 0:
                            st.session_state.px_to_mm_ratio = COIN_DIAMETER_MM / avg_px_diameter

            # Use existing scaling ratio if no new coin is detected
            px_to_mm_ratio = st.session_state.get("px_to_mm_ratio", None)

            # Draw OBB detections
            for detection in result.obb:
                # Get OBB detection info
                xywhr = detection.xywhr[0].cpu().numpy()
                class_id = int(detection.cls[0])
                confidence = float(detection.conf[0])
                class_name = CLASS_NAMES.get(class_id, f"Class {int(class_id)}")
                color = CATEGORY_COLORS.get(class_name, (0, 255, 0))
                
                # Generate unique ID for the object
                obj_id = generate_object_id(map(int, detection.xyxy[0]), class_name)
                
                # check whether tracked_objects are in session state
                if 'tracked_objects' not in st.session_state:
                    print("Initializing tracked_objects in session state")
                    # st.session_state.tracked_objects = {}
                    
    
                if obj_id not in st.session_state.tracked_objects: 
                    print(f"New object detected: {class_name} with ID: {obj_id}")
                    st.session_state.tracked_objects[obj_id] = class_name
                    print(f"Total tracked objects: {len(st.session_state.tracked_objects)}")

                if not SHOW_DETECTIONS:
                    continue
                
                # Convert xywhr to four corner points
                corners = xywhr_to_corners(xywhr)
                
                # Draw rotated rectangle
                for i in range(4):
                    start = tuple(corners[i].astype(int))
                    end = tuple(corners[(i + 1) % 4].astype(int))
                    cv2.line(img, start, end, color, 2)
                
                # Draw label
                label = f"CatID:{class_id} {confidence:.2f}"
                if class_id == COIN_CLASS_ID and px_to_mm_ratio:
                    diameter_px = (xywhr[2] + xywhr[3]) / 2
                    diameter_mm = diameter_px * px_to_mm_ratio
                    label += f", Dia: {diameter_mm:.2f}mm"
                elif class_id != COIN_CLASS_ID and px_to_mm_ratio:
                    length_px = max(xywhr[2], xywhr[3])
                    length_mm = length_px * px_to_mm_ratio
                    label += f", Length: {length_mm:.2f}mm"
                elif class_id != COIN_CLASS_ID:
                    label += ", Length: N/A (No Coin)"
                elif class_id == COIN_CLASS_ID:
                    label += ", Dia: N/A (No Ratio)"

                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Label background
                label_bg = (corners[0][0], corners[0][1] - text_height - 10,
                           corners[0][0] + text_width + 5, corners[0][1])
                cv2.rectangle(img, 
                            (int(label_bg[0]), int(label_bg[1])),
                            (int(label_bg[2]), int(label_bg[3])),
                            color, -1)
                
                # Label text
                cv2.putText(img, label, 
                          (int(corners[0][0] + 2), int(corners[0][1] - 5)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw orientation if enabled
                if SHOW_ORIENTATION:
                    center = (int(xywhr[0]), int(xywhr[1]))
                    endpoint = (int(center[0] + 20 * math.cos(xywhr[4])), 
                               int(center[1] + 20 * math.sin(xywhr[4])))
                    cv2.line(img, center, endpoint, (255, 255, 255), 2)
        print(f"The end of function--Total tracked objects: {len(st.session_state.tracked_objects)}")
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    except Exception as e:
        print(f"Processing error: {e}")
        return frame  # Return original frame if error occurs


    
def process_frame(frame, px_to_mm_ratio=None):
    """Process a single frame and return annotated image and detection data"""
    print("Processing frame...")
    results = model(frame, conf=CONFIDENCE_THRESHOLD)
    print(f"Model inference completed")
    if not results:
        print("No results found")
        return frame, [], px_to_mm_ratio
    print(f"Found {len(results)} results")
    result = results[0]
    filtered_detections = non_max_suppression(result.obb, IOU_THRESHOLD)
    print(f"Filtered {len(filtered_detections)} detections after NMS")
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
    # if two coin detected, use the higher confidence one
    if current_px_to_mm_ratio is None:
        highest_confidence = 0
        for detection in filtered_detections:
            if len(detection.cls) > 0 and int(detection.cls[0]) == COIN_CLASS_ID and len(detection.xywhr) > 0:
                confidence = detection.conf[0]
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    coin_xywhr = detection.xywhr[0].cpu().numpy()
                    width_px = coin_xywhr[2]
                    height_px = coin_xywhr[3]
                    avg_px_diameter = (width_px + height_px) / 2
                    if avg_px_diameter > 0:
                        current_px_to_mm_ratio = COIN_DIAMETER_MM / avg_px_diameter

    # Draw detections and track objects
    for detection in filtered_detections:
        if len(detection.cls) > 0 and len(detection.xywhr) > 0 and len(detection.xyxy) > 0:
            class_id = int(detection.cls[0])
            confidence = detection.conf[0]
            xywhr = detection.xywhr[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            class_name = CLASS_NAMES.get(class_id, f"Class {int(class_id)}")
            color = CATEGORY_COLORS.get(class_name, (0, 255, 0))

            # Generate unique ID for the object
            obj_id = generate_object_id((x1, y1, x2, y2), class_name)
            
            # Only count if not tracked before
            if obj_id not in st.session_state.tracked_objects:
                detected_objects.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(confidence),
                    "orientation": float(xywhr[4])
                })
                # Store the class name with the object ID
                st.session_state.tracked_objects[obj_id] = class_name

            label_text = f"CatID:{class_id} {confidence:.2f}"
            if class_id == COIN_CLASS_ID and current_px_to_mm_ratio:
                diameter_px = (xywhr[2] + xywhr[3]) / 2
                diameter_mm = diameter_px * current_px_to_mm_ratio
                label_text += f", Dia: {diameter_mm:.2f}mm"
            elif class_id != COIN_CLASS_ID and current_px_to_mm_ratio:
                length_px = max(xywhr[2], xywhr[3])
                length_mm = length_px * current_px_to_mm_ratio
                label_text += f", Length: {length_mm:.2f}mm"
            elif class_id != COIN_CLASS_ID:
                label_text += ", Length: N/A (No Coin)"
            elif class_id == COIN_CLASS_ID:
                label_text += ", Dia: N/A (No Ratio)"

            if SHOW_DETECTIONS:
                # Get OBB corners
                corners = xywhr_to_corners(xywhr)
                
                # Draw rotated rectangle
                for i in range(4):
                    start_point = tuple(corners[i])
                    end_point = tuple(corners[(i + 1) % 4])
                    draw.line([start_point, end_point], fill=color, width=BORDER_WIDTH)
                
                # Draw orientation indicator if enabled
                if SHOW_ORIENTATION:
                    center = (int(xywhr[0]), int(xywhr[1]))
                    endpoint = (int(center[0] + 20 * math.cos(xywhr[4])), 
                                int(center[1] + 20 * math.sin(xywhr[4])))
                    draw.line([center, endpoint], fill=(255, 255, 255), width=2)
                
                # Draw label
                text_width, text_height = get_text_size(draw, label_text, font)
                label_background = [(corners[0][0], corners[0][1] - text_height - 5),
                                  (corners[0][0] + text_width + 5, corners[0][1])]
                draw.rectangle(label_background, fill=color)
                draw.text((corners[0][0] + 2, corners[0][1] - text_height - 3), 
                          label_text, fill=(255, 255, 255), font=font)

    return np.array(pil_image), detected_objects, current_px_to_mm_ratio


def reset_detection_summary():
    st.session_state.tracked_objects = {}
    with summary_placeholder:
        st.empty()

# Main app
st.title("🔍 Screw Detection and Measurement (YOLOv11 OBB)")

# Create placeholders for content
frame_placeholder = st.empty()
main_content = st.container()  # Main content goes here

# Create placeholder at the bottom for summary
st.markdown("---")  # Add divider
summary_placeholder = st.container()  # Summary will be displayed here

def show_summary():
    """Display the detection summary with counts for each category"""
    with summary_placeholder:
            st.empty()

    if SHOW_SUMMARY:
        print("Showing summary...")
        # Check if tracked_objects exists in session state
        print("length of tracked_objects:", len(st.session_state.tracked_objects))
        if not hasattr(st.session_state, 'tracked_objects'):
            with summary_placeholder:
                st.warning("No tracking data available. Please start detection to see the summary.")
            return

        # Get all unique detected objects from session state
        if st.session_state.tracked_objects:
            # Filter out the coin class
            filtered_objects = {
                obj_id: class_name
                for obj_id, class_name in st.session_state.tracked_objects.items()
                if class_name != "10sen Coin"
            }
            
            # Count objects by class name
            class_counts = Counter(filtered_objects.values())
            
            summary_text = "### ✨ Unique Detections ✨\n"
            
            # Display total (excluding the coin)
            total_objects = len(filtered_objects)
            summary_text += f"- **Total unique objects detected (excluding coin): {total_objects}**\n\n"
            
            # Display count for each category
            summary_text += "#### Breakdown by Category:\n"
            for class_name, count in sorted(class_counts.items()):
                # Get color for this class
                color = CATEGORY_COLORS.get(class_name, (0, 255, 0))
                color_hex = "#{:02x}{:02x}{:02x}".format(*color)
                
                # Add colored category count
                summary_text += f"- <span style='color:{color_hex}'><b>{class_name}</b>: {count}</span>\n"
            
            with summary_placeholder:
                st.markdown(summary_text, unsafe_allow_html=True)
        else:
            with summary_placeholder:
                st.info("No objects detected yet.")

if input_method == "Upload Image":
    reset_detection_summary()  # Clear summary
    with main_content:
        st.subheader("Image Input")
        image_input_method = st.radio("Choose Input Method:", ("Upload", "Capture"))

        if image_input_method == "Upload":
            uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                frame = np.array(image)
            else:
                frame = None
        elif image_input_method == "Capture":
            captured_image = st.camera_input("Take a Picture")
            if captured_image is not None:
                frame = cv2.imdecode(np.frombuffer(captured_image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            else:
                frame = None

        if frame is not None:
            processed_frame, detected_objects, _ = process_frame(frame)
            st.image(processed_frame, channels="RGB")
            show_summary()

elif input_method == "Upload Video":
    reset_detection_summary()  # Clear summary
    with main_content:
        st.subheader("Video Input")
        video_input_method = st.radio("Choose Input Method:", ("Upload", "Capture"))

        if video_input_method == "Upload":
            uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
            if uploaded_video is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_video.read())
                video_path = tfile.name
            else:
                video_path = None
        elif video_input_method == "Capture":
            captured_video = st.camera_input("Record a Video")
            if captured_video is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(captured_video.getvalue())
                video_path = tfile.name
            else:
                video_path = None

        if video_path is not None:
            cap = cv2.VideoCapture(video_path)
            px_to_mm_ratio = None

            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Placeholders
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🎥 Input Video")
                original_frame_placeholder = st.empty()
            with col2:
                st.markdown("### 🛠️ Processed Video")
                processed_frame_placeholder = st.empty()

            progress_bar = st.progress(0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                processed_frame, _, px_to_mm_ratio = process_frame(frame, px_to_mm_ratio)

                # Display frames
                original_frame_placeholder.image(frame, channels="RGB", use_column_width=True)
                processed_frame_placeholder.image(processed_frame, channels="RGB", use_column_width=True)

                # Update progress
                current_frame += 1
                progress_bar.progress(min(current_frame / frame_count, 1.0))
                time.sleep(1 / fps)

            cap.release()
            show_summary()

elif input_method == "Webcam (Live Camera)":
    # reset_detection_summary()  # Clear summary
    with main_content:
        st.subheader("Live Camera Detection")
        webrtc_ctx = webrtc_streamer(
            key="screw-detection",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": WEBCAM_WIDTH},
                    "height": {"ideal": WEBCAM_HEIGHT},
                    "frameRate": {"ideal": 30},
                    "facingMode": {"ideal": "environment"}
                },
                "audio": False
            },
            async_processing=True
        )

        print(f" After frame ---Total tracked objects: {len(st.session_state.tracked_objects)}")
        # Periodically update the summary
        while webrtc_ctx.state.playing:
            print(f"In printing -- Total tracked objects: {len(st.session_state.tracked_objects)}")
            show_summary()  # Call the summary function to update the UI
            time.sleep(1)  # Update every second
