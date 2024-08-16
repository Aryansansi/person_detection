import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import os
import matplotlib.pyplot as plt

# Paths to the dataset directories
PATH = "data_file"
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

# Calculate the total number of images in each directory
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = len(os.listdir(test_dir))

# Model parameters
batch_size = 32
epochs = 20
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Image data generators with augmentation
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

validation_image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_image_generator = ImageDataGenerator(rescale=1./255)

# Data generators
train_data_gen = train_image_generator.flow_from_directory(
    PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    classes=['train'],
    class_mode="binary",
)

val_data_gen = validation_image_generator.flow_from_directory(
    PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    classes=['validation'],
    class_mode="binary",
)

test_data_gen = test_image_generator.flow_from_directory(
    PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    classes=['test'],
    class_mode="binary",
    shuffle=False
)

model_url = "rcnn"
detector = hub.load(model_url).signatures['default']

# Function to process images and get bounding boxes
def detect_objects(images):
    results = []
    for image in images:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # Convert to float32
        image = tf.expand_dims(image, axis=0)  # Add batch dimension
        
        # Ensure correct input type for model
        if image.dtype != tf.float32:
            image = tf.cast(image, tf.float32)
        
        result = detector(image)
        
        # Extract necessary components, handling missing keys
        boxes = result['detection_boxes'].numpy() if 'detection_boxes' in result else None
        scores = result['detection_scores'].numpy() if 'detection_scores' in result else None

        # Handle cases where boxes or scores are missing
        if boxes is None or scores is None or len(scores) == 0:
            results.append(((0, 0), 0))
            continue
        
        # Find the bounding box with the highest score
        best_idx = np.argmax(scores)
        box = boxes[best_idx]
        score = scores[best_idx]
        y_min, x_min, y_max, x_max = box
        
        height, width, _ = image.shape[1:]
        x_center = (x_min + x_max) * width / 2
        y_center = (y_min + y_max) * height / 2
        results.append(((x_center, y_center), score))
    
    return results

# Get center coordinates and scores for the test images
test_images, _ = next(test_data_gen)  # Get images and dummy labels
results = detect_objects(test_images)

# Function to plot images with bounding boxes
def plot_images_with_boxes(images, results):
    fig, axes = plt.subplots(len(images), 1, figsize=(5, len(images) * 3))
    for img, (coords, score), ax in zip(images, results, axes):
        ax.imshow(img)
        ax.axis('off')
        x_center, y_center = coords
        box_size = 20  # Define the size of the bounding box

        # Draw the bounding box
        rect = plt.Rectangle((x_center - box_size // 2, y_center - box_size // 2), box_size, box_size, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        ax.set_title(f"Center: ({x_center:.2f}, {y_center:.2f}), Score: {score:.2f}")
    plt.show()

plot_images_with_boxes(test_images[:5], results[:5])
