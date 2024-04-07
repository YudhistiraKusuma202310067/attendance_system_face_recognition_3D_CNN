import cv2
import os
from mtcnn import MTCNN
import shutil

def detect_faces_and_copy(input_folder, output_folder):
    # Inisialisasi detektor MTCNN
    detector = MTCNN()

    # Pastikan folder output ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop melalui setiap file gambar di folder input
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        # Baca gambar
        img = cv2.imread(file_path)

        # Deteksi wajah menggunakan MTCNN
        faces = detector.detect_faces(img)

        # Jika terdapat wajah yang terdeteksi, salin gambar ke folder output
        if faces:
            for i, face in enumerate(faces):
                # Dapatkan koordinat bounding box wajah
                x, y, w, h = face['box']

                # Potong gambar wajah
                face_img = img[y:y+h, x:x+w]

                # Simpan gambar wajah ke folder output
                output_file_path = os.path.join(output_folder, f"{file_name.split('.')[0]}_{i}.jpg")
                cv2.imwrite(output_file_path, face_img)
                print(f"Detected and saved face in {output_file_path}")

# Path folder input (dataset)
input_folder = 'dataset_testing'

# Path folder output untuk menyimpan foto wajah yang terdeteksi
output_folder = 'final_detected_face'

# Lakukan deteksi wajah dan salin gambar ke folder output
detect_faces_and_copy(input_folder, output_folder)
