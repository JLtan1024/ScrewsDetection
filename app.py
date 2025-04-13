import streamlit as st
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Set Paths - Update for Local Use
yolo_model_path = "yolo8_best.pt"
yolo_obb_model_path = data_root + "yolo11-obb.pt"
faster_rcnn_model_path = data_root + "faster_rcnn_screws.pth"

# Load YOLO Models
try:
    yolo_model = YOLO(yolo_model_path)
except Exception as e:
    st.error(f"Error loading YOLO models: {e}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to preprocess image
def preprocess_image(image):
    image = np.array(image)

    # Convert RGBA to RGB if necessary
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    return image

# Function to draw bounding boxes for Faster R-CNN predictions
def draw_boxes(image, prediction):
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    threshold = 0.5  # Confidence threshold
    image = image.copy()

    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            text = f"Class {label}: {score:.2f}"
            cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2, cv2.LINE_AA)

    return image

# Load Faster R-CNN Model
def get_faster_rcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Load trained Faster R-CNN model
num_classes = 14
faster_rcnn_model = get_faster_rcnn_model(num_classes)

try:
    faster_rcnn_model.load_state_dict(torch.load(faster_rcnn_model_path, map_location=device))
    faster_rcnn_model.to(device)
    faster_rcnn_model.eval()
except Exception as e:
    st.error(f"Error loading Faster R-CNN model: {e}")

# Streamlit UI
st.title("🔍 Screw Detection and Classification")

# Sidebar selection for input type
option = st.sidebar.radio("Choose Image Input Method", ("Upload an Image", "Take a Photo"))

# Image input: Either from file upload or camera
image = None
if option == "Upload an Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
elif option == "Take a Photo":
    camera_input = st.camera_input("Take a Picture")
    if camera_input is not None:
        image = Image.open(camera_input)

# If an image is provided, process it
if image:
    processed_image = preprocess_image(image)

    # YOLO Detection
    yolo_results = yolo_model(processed_image)
    yolo_obb_results = yolo_obb_model(processed_image)

    # Convert YOLO results for Streamlit display
    yolo_image = Image.fromarray(yolo_results[0].plot()[:, :, ::-1])  # Convert BGR to RGB
    yolo_obb_image = Image.fromarray(yolo_obb_results[0].plot()[:, :, ::-1])

    st.image(yolo_image, caption="YOLO v8 Detection", use_column_width=True)
    st.image(yolo_obb_image, caption="YOLO v11 OBB Detection", use_column_width=True)

    # Process with Faster R-CNN
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = faster_rcnn_model(image_tensor)[0]

    # Draw Faster R-CNN results
    faster_rcnn_image = draw_boxes(processed_image, predictions)

    # Convert to PIL Image for Streamlit display
    faster_rcnn_pil = Image.fromarray(cv2.cvtColor(faster_rcnn_image, cv2.COLOR_BGR2RGB))
    st.image(faster_rcnn_pil, caption="Faster R-CNN Detection", use_column_width=True)
