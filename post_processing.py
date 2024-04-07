import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models, Model
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from keras.models import load_model, Sequential
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Function to load the dataset for training
def load_dataset(dataset_folder):
    images = []
    labels = []
    
    # Create an empty mapping dictionary
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
            
    return np.array(images), np.array(labels), label_map

# Function to preprocess the images
def preprocess_images(images):
    # Convert to float and normalize
    images = images.astype('float32') / 255.0
    return images

# Load the dataset for training
training_images, training_labels, label_map = load_dataset("dataset")
training_images = preprocess_images(training_images)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(training_images, training_labels, test_size=0.2, random_state=42)

# # Define the CNN model architecture
# def create_model(input_shape, num_classes):
#     inputs = tf.keras.Input(input_shape)

#     x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
#     x = layers.MaxPool2D(2,2)(x)
#     x = layers.BatchNormalization()(x)

#     x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
#     x = layers.MaxPool2D(2,2)(x)
#     x = layers.BatchNormalization()(x)

#     x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
#     x = layers.MaxPool2D(2,2)(x)
#     x = layers.BatchNormalization()(x)

#     # x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
#     # x = layers.MaxPool2D(2,2)(x)
#     # x = layers.BatchNormalization()(x)

#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dense(units=256, activation="relu")(x)
#     x = layers.Dropout(0.3)(x)

#     outputs = layers.Dense(units=num_classes, activation="softmax")(x)

#     # Define the model
#     model = tf.keras.Model(inputs, outputs, name="2dcnn")
    
#     return model

# def create_model(input_shape, num_classes):
#     inputs = tf.keras.Input(input_shape)

#     x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(inputs)
#     x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
#     x = layers.MaxPool2D(2, 2)(x)

#     x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
#     x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
#     x = layers.MaxPool2D(2, 2)(x)

#     x = layers.Flatten()(x)
#     x = layers.Dense(units=128, activation="relu")(x)
#     x = layers.Dense(units=64, activation="relu")(x)
#     outputs = layers.Dense(units=num_classes, activation="softmax")(x)

#     model = tf.keras.Model(inputs, outputs, name="2dcnn")

#     return model

# def create_model(input_shape, num_classes):
#     inputs = layers.Input(input_shape)

#     x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
#     x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = layers.MaxPooling2D((2, 2))(x)

#     x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     x = layers.MaxPooling2D((2, 2))(x)

#     x = layers.Flatten()(x)
#     x = layers.Dense(128, activation='relu')(x)
#     x = layers.Dense(64, activation='relu')(x)
#     x = layers.Dense(num_classes)(x)
#     outputs = layers.Activation('softmax')(x)

#     model = Model(inputs, outputs)
#     return model

# def create_model(input_shape, num_classes):
#     inputs = layers.Input(input_shape)

#     x = layers.Conv2D(64, (3, 3), activation='relu')(inputs)
#     x = layers.Conv2D(64, (3, 3), activation='relu')(x)
#     x = layers.MaxPooling2D((2, 2))(x)

#     x = layers.Conv2D(128, (3, 3), activation='relu')(x)
#     x = layers.Conv2D(128, (3, 3), activation='relu')(x)
#     x = layers.MaxPooling2D((2, 2))(x)

#     x = layers.Flatten()(x)
#     x = layers.Dense(128, activation='relu')(x)
#     x = layers.Dense(64, activation='relu')(x)
#     x = layers.Dense(num_classes)(x)
#     outputs = layers.Activation('softmax')(x)

#     model = Model(inputs, outputs)
#     return model

def create_model(input_shape, num_classes):
    model = Sequential()

    model.add(layers.Conv2D(64, (3,3), padding='valid', activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3,3), padding='valid', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(128, (3,3), padding='valid', activation='relu'))
    model.add(layers.Conv2D(128, (3,3), padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation('softmax'))

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
history = model.fit(X_train, y_train, epochs=10, batch_size=32, 
                    validation_data=(X_val, y_val))

# Save the entire model to a single .h5 file
model_save_path = "model"
os.makedirs(model_save_path, exist_ok=True)
model.save(os.path.join(model_save_path, "face_model.h5"))

# Plotting training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plotting training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Predict validation set
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix
cm = confusion_matrix(y_val, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Membuat dictionary pemetaan antara label numerik dan nama kelas
label_to_class = {v: k for k, v in label_map.items()}

# Mengubah label numerik menjadi nama kelas
y_val_classes = [label_to_class[label] for label in y_val]
y_pred_classes = [label_to_class[label] for label in y_pred_classes]

# Classification report
print(classification_report(y_val_classes, y_pred_classes))