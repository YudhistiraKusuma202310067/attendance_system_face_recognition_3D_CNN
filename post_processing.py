import os
import random
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models

# Mendapatkan daftar nama folder mahasiswa di dalam folder "dataset"
mahasiswa_folders = os.listdir("dataset")

# Membuat folder untuk dataset testing
testing_folder = "dataset_testing"
os.makedirs(testing_folder, exist_ok=True)

# Inisialisasi list untuk menyimpan path file gambar yang dipilih
selected_images = []

# Iterasi melalui setiap folder mahasiswa
for mahasiswa_name in mahasiswa_folders:
    # Path ke folder mahasiswa
    mahasiswa_folder_path = os.path.join("dataset", mahasiswa_name)
    
    # Mendapatkan daftar gambar dari folder mahasiswa
    image_files = os.listdir(mahasiswa_folder_path)
    
    # Mengambil 10 gambar secara acak
    random_images = random.sample(image_files, 10)
    
    # Menambahkan path file gambar yang dipilih ke dalam list
    for image in random_images:
        selected_images.append(os.path.join(mahasiswa_folder_path, image))

# Mengacak urutan gambar
random.shuffle(selected_images)

# Menyalin gambar-gambar yang dipilih ke folder dataset testing dengan label 1 hingga 50
for idx, image_path in enumerate(selected_images, start=1):
    dst = os.path.join(testing_folder, f"{idx}.jpg")
    shutil.copyfile(image_path, dst)

print("Selesai Dataset_Testing")

print("Starting Training")
# Function to load the dataset for training
def load_dataset(dataset_folder):
    images = []
    labels = []
    
    # Create a mapping dictionary for folder names to integer labels
    label_map = {}
    for idx, folder_name in enumerate(os.listdir(dataset_folder)):
        label_map[folder_name] = idx
    
    for folder_name in os.listdir(dataset_folder):
        folder_path = os.path.join(dataset_folder, folder_name)
        label = label_map[folder_name]  # Get the integer label from the mapping
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            image = cv2.imread(img_path)
            # Resize the image to a fixed size (you can adjust this as needed)
            image = cv2.resize(image, (224, 224))
            images.append(image)
            labels.append(label)  # Use the integer label
    return np.array(images), np.array(labels)

# Function to preprocess the images
def preprocess_images(images):
    # Convert to float and normalize
    images = images.astype('float32') / 255.0
    return images

# Load the dataset for training
training_images, training_labels = load_dataset("dataset")
training_images = preprocess_images(training_images)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(training_images, training_labels, test_size=0.2, random_state=42)

# Define the CNN model architecture
def create_model(input_shape, num_classes):
    inputs = tf.keras.Input(input_shape)

    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool2D(2,2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool2D(2,2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool2D(2,2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool2D(2,2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=num_classes, activation="softmax")(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs, name="2dcnn")
    
    return model

# Define input shape and number of classes
input_shape = (224, 224, 3)  # Assuming images are resized to 224x224 RGB images
num_classes = len(os.listdir("dataset"))  # Number of classes is the number of folders in the dataset

# Create the CNN model
model = create_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=1, batch_size=32, 
                    validation_data=(X_val, y_val))

# Evaluate the model on the testing dataset
def load_testing_dataset(dataset_folder):
    images = []
    labels = []
    for img_name in os.listdir(dataset_folder):
        img_path = os.path.join(dataset_folder, img_name)
        image = cv2.imread(img_path)
        # Resize the image to match the input shape of the model
        image = cv2.resize(image, (224, 224))
        images.append(image)
        labels.append(int(img_name.split('.')[0]))  # Assuming image names are the labels
    return np.array(images), np.array(labels)

testing_images, testing_labels = load_testing_dataset("dataset_testing")
testing_images = preprocess_images(testing_images)

# Evaluate the model on the testing dataset
test_loss, test_accuracy = model.evaluate(testing_images, testing_labels)
print(f"Test Accuracy: {test_accuracy}")

# Save the entire model to a single .h5 file
model_save_path = "model"
os.makedirs(model_save_path, exist_ok=True)
model.save(os.path.join(model_save_path, "model.h5"))