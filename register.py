import cv2
import dlib
import os
import tkinter as tk
from tkinter import messagebox

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

# Mendapatkan jalur relatif ke file .dat
model_path = os.path.join("source_model", "shape_predictor_68_face_landmarks.dat")

# Inisialisasi detektor wajah Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

# Fungsi untuk menampilkan pesan konfirmasi dalam bentuk popup
def show_confirmation_message(prompt):
    root = tk.Tk()
    root.withdraw()
    return messagebox.askokcancel("Konfirmasi", prompt)

# Fungsi untuk mengambil dataset dengan konfirmasi
def capture_with_confirmation(prompt, count_start, count_limit):
    if not show_confirmation_message(prompt):
        return False
    
    count = count_start
    while count < count_start + count_limit:
        # Baca frame dari webcam
        ret, frame = cap.read()

        # Konversi ke grayscale untuk deteksi wajah
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah dengan Dlib
        faces = detector(gray)

        for face in faces:
            # Dapatkan landmarks wajah
            landmarks = predictor(gray, face)
            
            # Dapatkan koordinat dari landmarks tertentu
            left_eye = (landmarks.part(36).x, landmarks.part(36).y)
            right_eye = (landmarks.part(45).x, landmarks.part(45).y)
            top_of_forehead = (landmarks.part(19).x, landmarks.part(19).y)
            bottom_of_chin = (landmarks.part(8).x, landmarks.part(8).y)
            left_ear = (landmarks.part(0).x, landmarks.part(0).y)
            right_ear = (landmarks.part(16).x, landmarks.part(16).y)
            
            # Tentukan batas area yang akan dipotong
            x = left_ear[0]
            y = top_of_forehead[1]
            w = right_ear[0] - left_ear[0]
            h = bottom_of_chin[1] - top_of_forehead[1]

            # Perluas area untuk mencakup daerah rambut dan telinga
            x = max(0, x - 25)
            y = max(0, y - 100)
            w = min(frame.shape[1], w + 50)
            h = min(frame.shape[0], h + 100)

            # Potong wajah dari frame
            face_image = frame[y:y + h, x:x + w]

            # Ubah ukuran gambar menjadi 250x300 pixel
            face_image = cv2.resize(face_image, (224, 224))

            # Simpan gambar dengan label
            image_path = os.path.join(folder_path, f"{student_name}_{count}.png")
            cv2.imwrite(image_path, face_image)

            count += 1

            # Tampilkan frame dengan kotak deteksi wajah dan informasi count
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f'Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Tampilkan frame
        cv2.imshow('Capture Faces', frame)

        # Tunggu tombol yang ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return True

# Input nama mahasiswa
student_name = input("Masukkan nama mahasiswa: ")
folder_path = f"dataset/{student_name}"
os.makedirs(folder_path, exist_ok=True)

# Alur pengambilan gambar dengan konfirmasi
confirmation_prompts = [
    "Mohon posisikan diri Anda di depan kamera dengan sikap yang normal. Tekan 'OK' untuk melanjutkan.",
    "Mohon lepaskan atribut yang sedang Anda gunakan. Tekan 'OK' untuk melanjutkan.",
    "Mohon ubah ekspresi Anda. Tekan 'OK' untuk melanjutkan.",
    "Mohon hadapkan wajah ke arah serong kanan. Tekan 'OK' untuk melanjutkan.",
    "Mohon hadapkan wajah ke arah serong kiri. Tekan 'OK' untuk melanjutkan.",
]

# Inisialisasi variabel count_start
count_start = 0

for prompt in confirmation_prompts:
    if not capture_with_confirmation(prompt, count_start, 100):
        break
    count_start += 100

# Tutup webcam
cap.release()
cv2.destroyAllWindows()