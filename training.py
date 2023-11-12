import cv2
import os
import numpy as np
from skimage import exposure
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Fungsi callback untuk menampilkan progress
class ProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f'\nEpoch {epoch + 1}, Accuracy: {logs.get("accuracy")}, Loss: {logs.get("loss")}')

# Fungsi untuk memproses gambar
def preprocess_image(image):
    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Peningkatan kontras dengan equalisasi histogram
    enhanced_image = exposure.equalize_hist(gray)

    return enhanced_image

# Fungsi untuk membangun model 3D CNN
def build_cnn_model(input_shape):
    inputs = tf.keras.Input(input_shape)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = tf.keras.Model(inputs, outputs, name="3dcnn")
    return model

# Meminta pengguna untuk memasukkan nama mahasiswa yang ingin di-train
mahasiswa_name = input("Masukkan nama mahasiswa yang ingin di-train: ")

# Membaca data dan label untuk mahasiswa tertentu
data = []
labels = []

mahasiswa_folder_path = os.path.join("dataset", mahasiswa_name)
if os.path.exists(mahasiswa_folder_path):
    for filename in os.listdir(mahasiswa_folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(mahasiswa_folder_path, filename)
            original_image = cv2.imread(image_path)

            # Proses gambar
            preprocessed_image = preprocess_image(original_image)

            # Set label sesuai dengan nama folder mahasiswa
            labels.append(1)

    # Konversi menjadi numpy array
    labels = np.array(labels)

    # Pembagian data menjadi train dan test sets
    train_labels, test_labels = train_test_split(labels, test_size=0.2, random_state=42)

    # Membangun model CNN
    progress_callback = ProgressCallback()
    input_shape = (preprocessed_image.shape[0], preprocessed_image.shape[1], 1)
    model = build_cnn_model(input_shape)

    # Compile model.
    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["accuracy"],
    )

    # Define callbacks.
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "3d_image_classification.h5", save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15)

    # Train the model
    epochs = 10
    model.fit(
        np.zeros((len(train_labels), preprocessed_image.shape[0], preprocessed_image.shape[1], 1)),
        train_labels,
        epochs=epochs,
        batch_size=32,
        validation_data=(np.zeros((len(test_labels), preprocessed_image.shape[0], preprocessed_image.shape[1], 1)), test_labels),
        verbose=2,
        callbacks=[progress_callback, checkpoint_cb, early_stopping_cb],
    )

    # Evaluasi model pada data test
    predictions = model.predict(np.zeros((len(test_labels), preprocessed_image.shape[0], preprocessed_image.shape[1], 1)))
    predictions = (predictions > 0.5).astype(int)

    # Menampilkan laporan klasifikasi
    report = classification_report(test_labels, predictions)
    print(report)

else:
    print(f"Folder untuk mahasiswa dengan nama {mahasiswa_name} tidak ditemukan.")
