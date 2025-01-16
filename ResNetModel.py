import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from sklearn.model_selection import train_test_split
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

    with open(csv_path, "r") as file:
        lines = file.readlines()

    if lines[0].strip() == "image_path,label":
        lines = lines[1:]

    for line in lines:
        try:
            relative_path, label = line.strip().split(",")
            label = int(label)
            full_path = os.path.join(base_path, relative_path)
            image = Image.open(full_path).convert("RGB")
            image = image.resize(img_size)
            X.append(np.array(image))
            y.append(label)
        except Exception as e:
            print(f"Error processing line '{line.strip()}': {e}")

    X = np.array(X)
    y = np.array(y)

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

country = ["Algeria", "Angola", "DR Congo", "Egypt", "Ethiopia", "Ghana", "Kenya", "Namibia", "Nigeria", "South Africa"]

# Normalize pixel values (0-1 range)
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels
y_train_one_hot = to_categorical(y_train, num_classes=len(country))
y_test_one_hot = to_categorical(y_test, num_classes=len(country))

# Further split the training data for validation
X_train, X_val, y_train_one_hot, y_val_one_hot = train_test_split(
    X_train, y_train_one_hot, test_size=0.2, random_state=42
)

# Display a sample image
def plot_sample(X, y, index):
    plt.figure(figsize=(5, 5))
    plt.imshow(X[index].astype("uint8"))
    plt.title(f"Label: {country[np.argmax(y[index])]}", fontsize=14)
    plt.axis("off")
    plt.show()

plot_sample(X_train, y_train_one_hot, 40)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
datagen.fit(X_train)

# Load the ResNet50 model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze the base model initially

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(country), activation="softmax")
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Model summary
print(model.summary())

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train_one_hot, batch_size=16),
    epochs=20,
    validation_data=(X_val, y_val_one_hot),
)

# Unfreeze some layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Freeze all but the last 20 layers
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Fine-tune the model
history_fine = model.fit(
    datagen.flow(X_train, y_train_one_hot, batch_size=16),
    epochs=10,
    validation_data=(X_val, y_val_one_hot),
)

# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test_one_hot)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot training & validation metrics
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.plot(history_fine.history["accuracy"], label="Fine-Tuned Training Accuracy")
plt.plot(history_fine.history["val_accuracy"], label="Fine-Tuned Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.plot(history_fine.history["loss"], label="Fine-Tuned Training Loss")
plt.plot(history_fine.history["val_loss"], label="Fine-Tuned Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

# Visualize a test sample
plot_sample(X_test, y_test_one_hot, 40)
