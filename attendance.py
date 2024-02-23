import cv2
import numpy as np
from keras.models import load_model
import os

# Inisialisasi detektor wajah Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Ambang batas untuk pengenalan wajah
threshold = 0.5

# Path ke folder model
model_folder_path = "model"

# Memeriksa apakah folder model ada
if not os.path.exists(model_folder_path):
    raise FileNotFoundError(f"Folder model '{model_folder_path}' tidak ditemukan.")

# Membuat dictionary untuk menyimpan model yang dimuat
loaded_models = {}

# Memuat semua model dalam folder model
for model_file in os.listdir(model_folder_path):
    if model_file.endswith(".h5"):
        model_name = model_file[:-3]  # Menghapus ekstensi .h5
        model_path = os.path.join(model_folder_path, model_file)
        loaded_models[model_name] = load_model(model_path)

# Cek apakah setidaknya satu model telah dimuat
if not loaded_models:
    raise FileNotFoundError(f"Tidak ada file model (.h5) yang ditemukan dalam folder '{model_folder_path}'.")

# Dictionary untuk memetakan label ke nama mahasiswa
label_to_name = {}

# Ambil nama mahasiswa dari nama folder dataset
mahasiswa_folders = os.listdir("dataset")
for i, mahasiswa_name in enumerate(mahasiswa_folders):
    label_to_name[i] = mahasiswa_name

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()

    # Konversi frame ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah pada frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop melalui setiap wajah yang terdeteksi
    for (x, y, w, h) in faces:
        # Ambil ROI (Region of Interest) untuk wajah
        face_roi = gray[y:y + h, x:x + w]

        # Pra-pemrosesan gambar

        # Resize ROI menjadi ukuran yang sesuai dengan model (300x250)
        preprocessed_image = cv2.resize(face_roi, (300, 250))

        # Normalisasi nilai piksel
        preprocessed_image = preprocessed_image / 255.0

        # Bentuk ulang array agar sesuai dengan input model
        preprocessed_image = np.reshape(preprocessed_image, (1, 300, 250, 1))

        # Loop melalui semua model yang dimuat
        for model_name, model in loaded_models.items():
            # Lakukan klasifikasi menggunakan model
            prediction = model.predict(preprocessed_image)

            # Ambil indeks kelas dengan nilai probabilitas tertinggi sebagai label prediksi
            predicted_label = np.argmax(prediction)

            # Tentukan nama mahasiswa berdasarkan label prediksi
            nama_mahasiswa = label_to_name.get(predicted_label, "Mahasiswa Tidak Dikenali")

            # Tampilkan bounding box di sekitar wajah
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Tampilkan nama mahasiswa atau "Mahasiswa Tidak Dikenali" berdasarkan prediksi
            cv2.putText(frame, f"{model_name}: {nama_mahasiswa}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow('Face Recognition', frame)

    # Hentikan loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela
cap.release()
cv2.destroyAllWindows()
