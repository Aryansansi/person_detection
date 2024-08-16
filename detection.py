import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
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
IMG_HEIGHT = 150
IMG_WIDTH = 150
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

# Model definition with convolutional layers
model = models.Sequential([ 
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3)  
])

model.summary()

# Model compilation
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss='mean_squared_error',
              metrics=['accuracy'])
 
# Callbacks for learning rate reduction and early stopping
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000001)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Convert DirectoryIterator to tf.data.Dataset
def convert_to_tf_dataset(directory_iterator):
    def generator():
        for images, _ in directory_iterator:
            labels = np.random.rand(images.shape[0], 3)  # Replace with actual label data if available
            yield images, labels
    return tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32)))  # Output labels are now (x_center, y_center, score)

# Wrap data generators in tf.data.Dataset
train_dataset = convert_to_tf_dataset(train_data_gen).repeat()
val_dataset = convert_to_tf_dataset(val_data_gen).repeat()

# Check if weights file exists
weights_path = 'human_detection_model.weights.h5'
if os.path.exists(weights_path):
    model.load_weights(weights_path)
else:
    # Model training
    history = model.fit(train_dataset, steps_per_epoch=15, epochs=epochs,
                        validation_data=val_dataset, validation_steps=total_val // batch_size,
                        callbacks=[reduce_lr, early_stopping]) #steps_per_epoch=total_train // batch_size

    # Save model weights
    model.save_weights(weights_path)

    # Plot training results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
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

        max_score = -1
        max_coords = (0, 0)

        for i in range(grid_size):
            for j in range(grid_size):
                x = j * grid_width
                y = i * grid_height
                grid_block = image[y:y + grid_height, x:x + grid_width]
                
                # Resize grid block to match model input size
                grid_block = tf.image.resize(grid_block, (IMG_HEIGHT, IMG_WIDTH))
                grid_block = np.expand_dims(grid_block, axis=0)  # Add batch dimension
                
                # Predict (x_center, y_center, score)
                x_center, y_center, score = model.predict(grid_block)[0]

                # Adjust coordinates to be relative to the grid block
                x_center = x + (x_center * grid_width)
                y_center = y + (y_center * grid_height)

                if score > max_score:
                    max_score = score
                    max_coords = (x_center, y_center)

        results.append((max_coords, max_score))
    return results

# Get center coordinates and scores for the test images
results = process_image_grid(test_images)

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
