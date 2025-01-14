import os
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from keras.src.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Function to load dataset
def load_fage_custom_dataset(csv_path, base_path, img_size=(128, 128)):
    """
    Loads the FAGE dataset from a CSV-like format where each line contains an image path and label.

    Args:
        csv_path (str): Path to the CSV or text file containing image paths and labels.
        base_path (str): Base directory where the images are stored.
        img_size (tuple): Target size to resize images (width, height).

    Returns:
        (X_train, y_train), (X_test, y_test): Data split into training and testing sets.
    """
    X = []
    y = []

    # Load the dataset lines
    with open(csv_path, "r") as file:
        lines = file.readlines()

    # Skip the header line if it exists
    if lines[0].strip() == "image_path,label":
        lines = lines[1:]  # Exclude the first line

    for line in lines:
        try:
            # Split each line into path and label
            relative_path, label = line.strip().split(",")
            label = int(label)  # Convert label to an integer

            # Full path to the image
            full_path = os.path.join(base_path, relative_path)

            # Load and preprocess image
            image = Image.open(full_path).convert("RGB")
            image = image.resize(img_size)  # Resize to target size
            X.append(np.array(image))
            y.append(label)
        except Exception as e:
            print(f"Error processing line '{line.strip()}': {e}")

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split into Train/Test using directory-based logic
    train_idx = [i for i, line in enumerate(lines) if line.startswith("Training")]
    test_idx = [i for i, line in enumerate(lines) if line.startswith("Test")]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return (X_train, y_train), (X_test, y_test)


# Paths to dataset
csv_file = "ethnic_labels.csv"
base_directory = "FAGE"

# Load the dataset
(X_train, y_train), (X_test, y_test) = load_fage_custom_dataset(csv_file, base_directory)

# print(f"Training data: {X_train.shape}, {y_train.shape}")
# print(f"Testing data: {X_test.shape}, {y_test.shape}")
# print(f"y_train Testing data: {y_train[:10]}")

country = ["Algeria", "Angola", "DR Congo", "Egypt", "Ethiopia", "Ghana", "Kenya", "Namibia", "Nigeria", "South Africa"]

#Function to display image
def plot_sample(X, y, index):
    plt.figure(figsize=(5, 5))  # Set the figure size
    plt.imshow(X[index].astype("uint8"))  # Ensure the image data is displayed correctly
    plt.title(f"Label: {country[y[index]]}", fontsize=14)  # Display the label as the title
    plt.axis("off")  # Turn off the axis to focus on the image
    plt.show()


# Ensure labels are one-hot encoded
y_train_one_hot = to_categorical(y_train, num_classes=len(country))
y_test_one_hot = to_categorical(y_test, num_classes=len(country))

# Further split the training data for validation
X_train, X_val, y_train_one_hot, y_val_one_hot = train_test_split(
    X_train, y_train_one_hot, test_size=0.2, random_state=42
)

# Normalize pixel values (0-1 range)
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# CNN Model Architecture
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()

    # First convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third convolutional layer
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))  # To avoid overfitting
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


# Model configuration
input_shape = (128, 128, 3)
num_classes = len(country)

model = create_cnn_model(input_shape, num_classes)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Model Summary
print(model.summary())

# # Define the data augmentation configuration
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
#
# datagen.fit(X_train)

# Train the Model
history = model.fit(
    X_train,
    y_train_one_hot,
    batch_size=32,
    epochs=50,  # Adjust as needed
    validation_data=(X_val, y_val_one_hot),
)

# Evaluate on Test Data
test_loss, test_accuracy = model.evaluate(X_test, y_test_one_hot)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


plt.figure(figsize=(12, 5))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

plot_sample(X_test, y_test, 400)