import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.callbacks import LearningRateScheduler # type: ignore
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
GRID_SIZE = 4  

# Image data generators with augmentation
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2]
)

validation_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

# Data generators
train_data_gen = train_image_generator.flow_from_directory(
    PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    classes=['train'],
    class_mode=None
)

val_data_gen = validation_image_generator.flow_from_directory(
    PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    classes=['validation'],
    class_mode=None
)

test_data_gen = test_image_generator.flow_from_directory(
    PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    classes=['test'],
    class_mode=None,
    shuffle=False
)

# Model definition using VGG16 as the base model
base_model = VGG16(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                   include_top=False,
                   weights='imagenet')
base_model.trainable = False  # Freeze the base model

model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5)  
])

model.build((None, IMG_HEIGHT, IMG_WIDTH, 3))  # Explicitly define the input shape
model.summary()

# Model compilation
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='mean_squared_error',
              metrics=['accuracy'])

# Callbacks for learning rate reduction and early stopping
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

def lr_schedule(epoch,lr):
    if epoch > 10:
        lr = lr * 0.5
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

# Convert DirectoryIterator to tf.data.Dataset
def convert_to_tf_dataset(directory_iterator):
    def generator():
        for images in directory_iterator:
            labels = np.random.rand(images.shape[0], 5)  # Adjusted to 5 outputs
            yield images, labels
    return tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 5), dtype=tf.float32)))  # Ensure labels shape matches model output

# Wrap data generators in tf.data.Dataset
train_dataset = convert_to_tf_dataset(train_data_gen).repeat()
val_dataset = convert_to_tf_dataset(val_data_gen).repeat()

# Check if weights file exists
weights_path = 'human_detection_vgg16.weights.h5'
if os.path.exists(weights_path):
    model.load_weights(weights_path)
else:
    # Model training
    history = model.fit(train_dataset, steps_per_epoch=15, epochs=epochs,
                        validation_data=val_dataset, validation_steps=total_val // batch_size,
                        callbacks=[reduce_lr, early_stopping, lr_scheduler])

    # Save model weights
    model.save_weights(weights_path)

    # Plot training results
    acc = history.history['accuracy']
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history['loss']
    val_loss = history.history.get('val_loss', [])
    epochs_range = range(len(acc))

    def plot_learning_curves(history):
        acc = history.history['accuracy']
        val_acc = history.history.get('val_accuracy', [])
        loss = history.history['loss']
        val_loss = history.history.get('val_loss', [])
        epochs_range = range(len(acc))

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    plot_learning_curves(history)
    
# Evaluate the model on the test data
test_images, _ = next(test_data_gen)  # Get images and dummy labels
predictions = model.predict(test_data_gen)

# Function to process images in a grid format and find bounding box
def process_image_grid(images, grid_size=GRID_SIZE):
    results = []
    for image in images:
        height, width, _ = image.shape
        grid_height = height // grid_size
        grid_width = width // grid_size

        best_box = None
        max_score = -1

        for i in range(grid_size):
            for j in range(grid_size):
                x = j * grid_width
                y = i * grid_height
                grid_block = image[y:y + grid_height, x:x + grid_width]

                # Resize grid block to match model input size
                grid_block = tf.image.resize(grid_block, (IMG_HEIGHT, IMG_WIDTH))
                grid_block = np.expand_dims(grid_block, axis=0)  # Add batch dimension
                
                # Predict bounding box coordinates and score
                x1, y1, x2, y2, score = model.predict(grid_block)[0]

                # Adjust coordinates to be relative to the grid block
                x1 = x + (x1 * grid_width)
                y1 = y + (y1 * grid_height)
                x2 = x + (x2 * grid_width)
                y2 = y + (y2 * grid_height)

                # Ensure coordinates are within image bounds
                x1 = max(0, min(width, x1))
                y1 = max(0, min(height, y1))
                x2 = max(0, min(width, x2))
                y2 = max(0, min(height, y2))

                if score > max_score:
                    max_score = score
                    best_box = (x1, y1, x2, y2)

        results.append((best_box, max_score))
    return results

# Get center coordinates and scores for the test images
results = process_image_grid(test_images)

# Function to plot images with bounding boxes
def plot_images_with_boxes(images, results):
    fig, axes = plt.subplots(len(images), 1, figsize=(5, len(images) * 3))
    for img, (box, score), ax in zip(images, results, axes):
        ax.imshow(img)
        ax.axis('off')
        x1, y1, x2, y2 = box
        box_size = 2  # Define the thickness of the bounding box line

        # Draw the bounding box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=box_size)
        ax.add_patch(rect)
        ax.set_title(f"Box: ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f}), Score: {score:.2f}")
    plt.show()

plot_images_with_boxes(test_images[:5], results[:5])
