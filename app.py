import cv2
import av
import torch
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# Load the YOLOv8/YOLOv11 model
# Replace with your model path or use a pretrained one
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # or yolov8, yolov11, etc.
model.conf = 0.5  # confidence threshold

# Define a label-color map
def get_color(label):
    np.random.seed(hash(label) % 12345)
    return tuple(int(x) for x in np.random.randint(0, 255, size=3))

# Callback for video frame processing
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # Inference
    results = model(img)
    for *box, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]
        color = get_color(label)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("YOLO Real-Time Object Detection")
st.markdown("Live object detection on webcam feed using YOLO and streamlit-webrtc.")

webrtc_ctx = webrtc_streamer(
    key="yolo-live-detect",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)