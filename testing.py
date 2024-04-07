import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model
import matplotlib.pyplot as plt

# Path folder dataset
dataset_dir = 'dataset'
dataset_testing_dir = 'dataset_testing'
dataset_testing_group_dir = 'dataset_testing_group'

# Path model yang telah dilatih
model_path = 'model/face_model.h5'

# Daftar nama kelas
# class_names = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]
class_names = ['Febri_Fairuz', 'Michael_Mervin_Ruswan', 'Muhammad_Ilham', 'Raihan_Dwi_Pratama', 'Raihan_Dwi_Win_Cahya', 'Raiyana_Jan_Winata']
print("Class Names:", class_names)

# Memuat model pengenalan wajah
face_recognition_model = load_model(model_path, compile=False)

# Inisialisasi detektor MTCNN
detector = MTCNN()

def detect_and_segment_faces(image_path):
    img = cv2.imread(image_path)

    # Deteksi wajah menggunakan MTCNN
    faces = detector.detect_faces(img)

    for face in faces:
        x, y, w, h = face['box']
        face_image = img[y:y+h, x:x+w]
        face_image = cv2.resize(face_image, (50, 50))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        # Tampilkan gambar face_image
        # plt.imshow(face_image)
        # plt.axis('off')
        # plt.show()

        # Prediksi kelas wajah
        prediction = predict_face_label(face_image)
        max_prob = np.max(prediction)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]

        # Gambarkan kotak di sekitar wajah pada gambar asli
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Tampilkan prediksi label dan akurasi pada gambar asli
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (x, y - 10)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        label_text = f"{predicted_class_name} ({max_prob*100:.2f}%)"
        img = cv2.putText(img, label_text, org, font, fontScale, color, thickness, cv2.LINE_AA)

        print(f"Prediksi: {predicted_class_name} Akurasi {max_prob*100:.2f}%")

    # Tampilkan gambar asli dengan kotak di sekitar wajah dan label prediksi
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def predict_face_label(face_image):
    # Normalisasi gambar wajah
    face_image = np.expand_dims(face_image, axis=0)
    face_image = face_image / 255.0

    # Prediksi kelas wajah
    prediction = face_recognition_model.predict(face_image)

    return prediction

# Loop melalui setiap gambar di folder dataset testing single
for filename in os.listdir(dataset_testing_dir):
    image_path = os.path.join(dataset_testing_dir, filename)
    detect_and_segment_faces(image_path)

# Loop melalui setiap gambar di folder dataset testing group
# for filename in os.listdir(dataset_testing_group_dir):
#     image_path = os.path.join(dataset_testing_group_dir, filename)
#     detect_and_segment_faces(image_path)
    
# # Set path gambar yang ingin Anda uji
# image_path = 'dataset_testing/4.jpeg'
# detect_and_segment_faces(image_path)