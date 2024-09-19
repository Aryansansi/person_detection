# Person Detection Project

This repository contains three main Python scripts, each designed for specific object detection tasks. The models implemented utilize various deep learning frameworks and techniques.

## 1. `detection.py`

### Description:
This script implements an object detection model using TensorFlow's Keras API. The model is designed to detect and localize objects within images by predicting the center coordinates and a confidence score.

### Key Features:
- **Custom Convolutional Neural Network (CNN):** Built from scratch with layers including Conv2D, MaxPooling, Dense, and Dropout.
- **Data Augmentation:** Extensive augmentation techniques like rotation, width and height shift, zoom, and shear are applied to improve model robustness.
- **Training and Evaluation:** The model is trained on the provided dataset, and training/validation accuracy and loss are plotted for performance visualization.
- **Bounding Box Prediction:** The script includes a method for processing images in a grid format to locate the most probable bounding box.

### Usage:
To run the detection model, execute the script:
```bash
python detection.py
```

Ensure the data is placed in the `data_file` directory, structured with subdirectories for `train`, `validation`, and `test`.

## 2. `main_rcnn.py`

### Description:
This script utilizes a pre-trained Region-based Convolutional Neural Network (RCNN) from TensorFlow Hub to perform object detection. The model outputs bounding boxes and associated confidence scores for objects detected in images.

### Key Features:
- **Pre-trained Model:** Leverages a pre-trained RCNN from TensorFlow Hub for efficient object detection.
- **Bounding Box Detection:** Detects and outputs bounding boxes with associated scores.
- **Data Augmentation:** Similar to the `detection.py` script, data augmentation is applied to the input images.

### Usage:
To run the RCNN-based detection model, execute the script:
```bash
python main_rcnn.py
```

Ensure the data is placed in the `data_file` directory, structured with subdirectories for `train`, `validation`, and `test`.

## 3. `person_detection.py`

### Description:
This script builds on the VGG16 architecture, a pre-trained model from the ImageNet dataset, to detect and localize objects in images. The model is customized for detecting persons within images, predicting bounding box coordinates and scores.

### Key Features:
- **VGG16-based Model:** Utilizes the VGG16 architecture for feature extraction, followed by custom Dense layers for object detection.
- **Learning Rate Scheduler:** Implements a learning rate scheduler for better convergence during training.
- **Bounding Box Prediction:** Similar to the `detection.py` script, the model predicts bounding boxes and scores for the detected objects.

### Usage:
To run the person detection model, execute the script:
```bash
python person_detection.py
```

Ensure the data is placed in the `data_file` directory, structured with subdirectories for `train`, `validation`, and `test`.

## Dataset Structure

Ensure that your dataset is structured as follows:
```
data_file/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── validation/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

## Dependencies

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- TensorFlow Hub (for `main_rcnn.py`)

To install the required packages, run:
```bash
pip install tensorflow numpy matplotlib tensorflow-hub
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

Special thanks to the open-source community and contributors to TensorFlow, Keras, and other associated libraries used in this project.
