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
def build_cnn_model(input_shape, num_classes):
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

    # x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    # x = layers.MaxPool3D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPool3D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPool3D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPool3D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.GlobalAveragePooling3D()(x)
    # x = layers.Dense(units=512, activation="relu")(x)
    # x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=num_classes, activation="softmax")(x)

    # Define the model.
    model = tf.keras.Model(inputs, outputs, name="2dcnn")
    return model


# Meminta pengguna untuk memasukkan nama mahasiswa yang ingin di-train
mahasiswa_name = input("Masukkan nama mahasiswa yang ingin di-train: ")

# Membaca data dan label untuk mahasiswa tertentu
data = []
labels = []
mahasiswa_names = []

# Dictionary untuk menyimpan nama mahasiswa beserta labelnya
mahasiswa_label_dict = {}

mahasiswa_folder_path = os.path.join("dataset", mahasiswa_name)
if os.path.exists(mahasiswa_folder_path):
    for filename in os.listdir(mahasiswa_folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(mahasiswa_folder_path, filename)
            original_image = cv2.imread(image_path)

            # Proses gambar
            preprocessed_image = preprocess_image(original_image)

            # Set label sesuai dengan nama folder mahasiswa
            labels.append(0)  # Mengubah label menjadi 0
            mahasiswa_names.append(mahasiswa_name)

    # Konversi menjadi numpy array
    labels = np.array(labels)

    # Membagi data menjadi train dan test sets
    train_labels, test_labels = train_test_split(labels, test_size=0.2, random_state=42)

    # Membangun model CNN
    progress_callback = ProgressCallback()
    input_shape = (preprocessed_image.shape[0], preprocessed_image.shape[1], 1)
    num_classes = len(np.unique(labels))  # Menentukan jumlah kelas
    model = build_cnn_model(input_shape, num_classes)

    # Compile model.
    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["accuracy"],
    )

    # Path untuk folder model
    model_folder_path = "model"

    # Pastikan folder model ada atau buat jika belum ada
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    # Define callbacks untuk menyimpan model ke dalam folder model
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        os.path.join(model_folder_path, f"{mahasiswa_name}_2d_image_classification.h5"),
        save_best_only=True
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
    predictions = np.argmax(predictions, axis=1)

    # Menampilkan laporan klasifikasi
    report = classification_report(test_labels, predictions)
    print(report)

else:
    print(f"Folder untuk mahasiswa dengan nama {mahasiswa_name} tidak ditemukan.")


# import cv2
# import os
# import numpy as np
# from skimage import exposure
# import tensorflow as tf
# from tensorflow import keras
# from keras import layers, models
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# # Fungsi callback untuk menampilkan progress
# class ProgressCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         print(f'\nEpoch {epoch + 1}, Accuracy: {logs.get("accuracy")}, Loss: {logs.get("loss")}')

# # Fungsi untuk memproses gambar
# def preprocess_image(image):
#     # Konversi gambar ke grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Peningkatan kontras dengan equalisasi histogram
#     enhanced_image = exposure.equalize_hist(gray)

#     return enhanced_image

# # Fungsi untuk membangun model 3D CNN
# def build_cnn_model(input_shape, num_classes):
#     inputs = tf.keras.Input(input_shape)

#     x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
#     x = layers.MaxPool3D(pool_size=2)(x)
#     x = layers.BatchNormalization()(x)

#     x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
#     x = layers.MaxPool3D(pool_size=2)(x)
#     x = layers.BatchNormalization()(x)

#     x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
#     x = layers.MaxPool3D(pool_size=2)(x)
#     x = layers.BatchNormalization()(x)

#     x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
#     x = layers.MaxPool3D(pool_size=2)(x)
#     x = layers.BatchNormalization()(x)

#     x = layers.GlobalAveragePooling3D()(x)
#     x = layers.Dense(units=512, activation="relu")(x)
#     x = layers.Dropout(0.3)(x)

#     outputs = layers.Dense(units=num_classes, activation="softmax")(x)

#     # Define the model.
#     model = tf.keras.Model(inputs, outputs, name="3dcnn")
#     return model

# # Meminta pengguna untuk memasukkan nama mahasiswa yang ingin di-train
# mahasiswa_name = input("Masukkan nama mahasiswa yang ingin di-train: ")

# # Membaca data dan label untuk mahasiswa tertentu
# data = []
# labels = []
# mahasiswa_names = []

# # Dictionary untuk menyimpan nama mahasiswa beserta labelnya
# mahasiswa_label_dict = {}

# mahasiswa_folder_path = os.path.join("dataset", mahasiswa_name)
# if os.path.exists(mahasiswa_folder_path):
#     for filename in os.listdir(mahasiswa_folder_path):
#         if filename.endswith(".png"):
#             image_path = os.path.join(mahasiswa_folder_path, filename)
#             original_image = cv2.imread(image_path)

#             # Proses gambar
#             preprocessed_image = preprocess_image(original_image)

#             # Tambahkan dimensi kedalaman tambahan
#             preprocessed_image = np.expand_dims(preprocessed_image, axis=-1)

#             # Pastikan bentuk gambar yang telah diproses
#             print("Bentuk gambar yang diproses:", preprocessed_image.shape)

#             # Tambahkan gambar ke dalam data
#             data.append(preprocessed_image)

#             # Set label sesuai dengan nama folder mahasiswa
#             labels.append(0)  # Mengubah label menjadi 0
#             mahasiswa_names.append(mahasiswa_name)

#     # Konversi menjadi numpy array
#     data = np.array(data)
#     labels = np.array(labels)

#     # Membagi data menjadi train dan test sets
#     train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

#     # Membangun model CNN
#     progress_callback = ProgressCallback()
#     input_shape = train_data.shape[1:]  # Menggunakan bentuk gambar sebagai input_shape, kecuali batch_size
#     num_classes = len(np.unique(labels))  # Menentukan jumlah kelas
#     model = build_cnn_model(input_shape, num_classes)

#     # Compile model.
#     initial_learning_rate = 0.0001
#     lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#         initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
#     )
#     model.compile(
#         loss="sparse_categorical_crossentropy",
#         optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
#         metrics=["accuracy"],
#     )

#     # Path untuk folder model
#     model_folder_path = "model"

#     # Pastikan folder model ada atau buat jika belum ada
#     if not os.path.exists(model_folder_path):
#         os.makedirs(model_folder_path)

#     # Define callbacks untuk menyimpan model ke dalam folder model
#     checkpoint_cb = keras.callbacks.ModelCheckpoint(
#         os.path.join(model_folder_path, f"{mahasiswa_name}_3d_image_classification.h5"),
#         save_best_only=True
#     )
#     early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15)

#     # Train the model
#     epochs = 10
#     model.fit(
#         np.repeat(np.expand_dims(preprocessed_image, axis=0), len(train_labels), axis=0),
#         train_labels,
#         epochs=epochs,
#         batch_size=32,
#         validation_data=(np.repeat(np.expand_dims(preprocessed_image, axis=0), len(test_labels), axis=0), test_labels),
#         verbose=2,
#         callbacks=[progress_callback, checkpoint_cb, early_stopping_cb],
#     )

#     # Evaluasi model pada data test
#     predictions = model.predict(np.repeat(np.expand_dims(preprocessed_image, axis=0), len(test_labels), axis=0))
#     predictions = np.argmax(predictions, axis=1)

#     # Menampilkan laporan klasifikasi
#     report = classification_report(test_labels, predictions)
#     print(report)

# else:
#     print(f"Folder untuk mahasiswa dengan nama {mahasiswa_name} tidak ditemukan.")
