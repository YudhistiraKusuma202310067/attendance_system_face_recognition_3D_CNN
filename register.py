import cv2
import dlib
import os

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

# Inisialisasi detektor wajah Dlib
detector = dlib.get_frontal_face_detector()

# Buat folder untuk menyimpan gambar
student_name = input("Masukkan nama mahasiswa: ")
folder_path = f"dataset/{student_name}"
os.makedirs(folder_path, exist_ok=True)

count = 0

while count < 400:
    # Baca frame dari webcam
    ret, frame = cap.read()

    # Konversi ke grayscale untuk deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dengan Dlib
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Potong wajah dari frame
        face_image = frame[y:y + h, x:x + w]

        # Ubah ukuran gambar menjadi 250x250 pixel
        face_image = cv2.resize(face_image, (250, 250))

        # Simpan gambar dengan label
        image_path = os.path.join(folder_path, f"{student_name}_{count}.png")
        cv2.imwrite(image_path, face_image)

        count += 1

        # Tampilkan frame dengan kotak deteksi wajah dan informasi count
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Tampilkan frame
    cv2.imshow('Capture Faces', frame)

    # Hentikan loop jika pengguna menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup webcam dan jendela tampilan
cap.release()
cv2.destroyAllWindows()