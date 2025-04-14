import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Set page config
st.set_page_config(page_title="🔍 Screw Detection", layout="wide")

# Load YOLO Model
@st.cache_resource
def load_model():
    try:
        model = YOLO("yolo8_best.pt")  # Update with your model path
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

# Function to preprocess image
def preprocess_image(image):
    image = np.array(image)
    # Convert RGBA to RGB if necessary
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image

# Streamlit UI
st.title("🔍 Screw Detection and Classification")

# Sidebar for additional options
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5,
        help="Adjust the minimum confidence level for detections"
    )
    
    # Choose input method
    input_method = st.radio(
        "Select Input Method",
        ("Upload Image", "Use Camera"),
        index=0
    )

# Image input based on selection
image = None
if input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "png", "jpeg"],
        help="Upload an image of screws for detection"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
else:
    camera_input = st.camera_input("Take a picture of screws")
    if camera_input is not None:
        image = Image.open(camera_input)

# Process image if available
if image:
    # Display original image
    st.subheader("Original Image")
    st.image(image, use_column_width=True)
    
    # Load model (cached)
    model = load_model()
    
    if model is not None:
        # Preprocess and predict
        processed_image = preprocess_image(image)
        
        # Make predictions
        with st.spinner("Detecting screws..."):
            results = model(processed_image, conf=confidence_threshold)
            
            # Plot results
            plotted_image = results[0].plot()[:, :, ::-1]  # Convert BGR to RGB
            
            # Display results
            st.subheader("Detection Results")
            st.image(plotted_image, use_column_width=True)
            
            # Show detection details
            st.subheader("Detection Details")
            
            # Get detection information
            boxes = results[0].boxes
            if len(boxes) > 0:
                # Create a table with detection info
                detection_data = []
                for i, box in enumerate(boxes):
                    detection_data.append({
                        "Screw #": i+1,
                        "Class": model.names[int(box.cls)],
                        "Confidence": f"{float(box.conf):.2f}",
                        "Position": f"({int(box.xyxy[0][0])}, {int(box.xyxy[0][1])})"
                    })
                
                # Display as table
                st.table(detection_data)
                
                # Summary statistics
                st.write(f"**Total screws detected:** {len(boxes)}")
            else:
                st.warning("No screws detected in the image.")
    else:
        st.error("Model failed to load. Please check the model file.")
