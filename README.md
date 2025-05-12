Screw Detection, Classification & Measurement using Object Detection Models
This project involves training and evaluating multiple object detection models—including Mask R-CNN, Faster R-CNN, YOLOv8-OBB, and YOLOv11-OBB—for detecting and classifying screws, nuts, and bolts. Some models are also fine-tuned to detect a reference coin to enable real-world length estimation of screws.

🔗 Demo
Try the live detection and measurement app here:
👉 Streamlit Deployment
https://fasternerdetection-cvyemai8u2v5npjsqccatb.streamlit.app/

📁 Dataset Preparation
All data preparation steps are provided in the loadData.ipynb notebook. Please refer to that notebook for the complete preprocessing and dataset creation pipeline before training.

Datasets
datasets/RCNN_YOLO_SET:
Uploaded as mvtec-screws/RCNN_YOLO_SET in the Kaggle input directory.
Used for Faster R-CNN, YOLOv8, and YOLOv11 training.

COCO-format annotations (with segmentation attributes) used for Mask R-CNN training are available under the Kaggle dataset:
coco-screw-with-seg
Files:

mvtec_screws_train_with_seg.json

mvtec_screws_val_with_seg.json

mvtec_screws_test_with_seg.json

🧪 Model Training
All training is conducted on Kaggle Notebooks using T4 GPUs.

Notebooks for each model architecture are available under shared resources in the Kaggle group (see below).

Available Training Notebooks
MRCNN – Mask R-CNN with segmentation.

FASTER_RCNN – Two-stage object detector.

yolo8_obb – YOLOv8 with oriented bounding boxes.

yolo11_obb – YOLOv11 with oriented bounding boxes.

yolo11-coin-added – YOLOv11 trained with coin detection for screw length estimation.

📊 Results
Evaluation results are available in the Kaggle group under the Results section. These include performance metrics across all models, screw types, and test configurations.

✅ Pre-requisite
You must join the Kaggle group to access datasets and notebooks.

Join the Group
Sign up at Kaggle if you don’t have an account.

Click the link below to request access to the group:
👉 Join Kaggle Group

Once joined, you'll get access to:
✅ Training Notebooks

✅ Result Reports

✅ All Datasets (MVTEC-SCREWS, COCO_screw_with_seg)
