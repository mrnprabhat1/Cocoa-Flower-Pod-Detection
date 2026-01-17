# Cocoa Flower and Cocoa Pod Detection Using Deep Learning

## Project Overview
This repository presents a deep learning based object detection pipeline for identifying cocoa flowers and cocoa pods in natural field images.  
The objective of this project is to demonstrate the feasibility of applying modern object detection techniques to agricultural monitoring tasks under limited data conditions.

The project emphasizes two main components.  
Data augmentation with label consistency and deep learning based object detection using a two stage detector.

## Key Contributions
- Label consistent data augmentation for object detection
- Robust handling of Pascal VOC XML annotations
- Bounding box validation and filtering
- Training of a Faster R CNN based detector
- IoU based quantitative evaluation
- Visualization of ground truth and predictions

## Dataset Description
- Data type: RGB field images
- Original images: 18
- Augmented images: 54
- Total dataset size: 72 images
- Object classes:
  - Flower
  - Cocoa pod
- Annotation format: Pascal VOC XML

Images were collected under real field conditions and manually annotated.  
Due to the limited dataset size, data augmentation plays a critical role in model training.

## Data Augmentation
A comprehensive data augmentation pipeline is implemented to increase dataset diversity while preserving annotation correctness.

Augmentation techniques include:
- Geometric transformations such as flip, rotation, scaling, and translation
- Color and lighting variation including brightness, contrast, hue, and saturation
- Noise and blur effects to simulate real world conditions
- Random crop followed by resizing

Bounding boxes are transformed together with images.  
Invalid or extremely small bounding boxes are automatically filtered.

## Model Architecture
- Detection framework: Faster R CNN
- Backbone network: ResNet50 with Feature Pyramid Network
- Pretrained weights: COCO dataset
- Detection head customized for two object classes and background

The two stage architecture is selected to prioritize detection accuracy over inference speed.

## Training Configuration
- Framework: PyTorch and Torchvision
- Optimizer: Stochastic Gradient Descent
- Learning rate: 0.005
- Batch size: 2
- Training epochs: 100
- Execution environment: Google Colab GPU

## Evaluation Methodology
Model performance is evaluated on the validation dataset using an IoU based matching strategy.

Evaluation details:
- Confidence threshold: 0.7
- IoU threshold: 0.7
- Metrics reported:
  - Precision
  - Recall
  - F1 score

Standard mean average precision is not reported due to the small dataset size.  
The selected metrics provide interpretable performance evaluation under strict localization constraints.

## Project Structure
Example augmented data structure:

