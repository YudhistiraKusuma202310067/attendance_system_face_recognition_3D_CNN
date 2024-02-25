import os
import cv2
import random
import shutil

# Fungsi untuk melakukan flipping pada gambar
def flip_image(image):
    return cv2.flip(image, 1)

# Fungsi untuk melakukan scaling pada gambar
def scale_image(image, scale_factor):
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

# Fungsi untuk melakukan cropping pada gambar
def crop_image(image, x, y, w, h):
    return image[y:y+h, x:x+w]

# Fungsi untuk melakukan mirroring pada gambar
def mirror_image(image):
    return cv2.flip(image, 0)

# Fungsi untuk melakukan rotasi pada gambar
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

# Fungsi untuk memilih gambar acak dari dataset
def select_random_image(dataset_folder):
    images = os.listdir(dataset_folder)
    random_image = random.choice(images)
    image_path = os.path.join(dataset_folder, random_image)
    return cv2.imread(image_path)

# Fungsi untuk melakukan augmentasi pada gambar dan menyimpan gambar-gambar hasil augmentasi
def augment_dataset(student_name):
    dataset_folder = os.path.join("dataset", student_name)
    augmented_folder = os.path.join("dataset", "augmented_dataset")

    os.makedirs(augmented_folder, exist_ok=True)
    for i in range(100):
        image = select_random_image(dataset_folder)
        
        flipped_image = flip_image(image)
        cv2.imwrite(os.path.join(augmented_folder, f"{student_name}_flipped_{i}.png"), flipped_image)
        
        scaled_image = scale_image(image, random.uniform(0.5, 2.0))
        cv2.imwrite(os.path.join(augmented_folder, f"{student_name}_scaled_{i}.png"), scaled_image)
        
        cropped_image = crop_image(image, random.randint(0, image.shape[1]//2), random.randint(0, image.shape[0]//2), random.randint(image.shape[1]//2, image.shape[1]), random.randint(image.shape[0]//2, image.shape[0]))
        cv2.imwrite(os.path.join(augmented_folder, f"{student_name}_cropped_{i}.png"), cropped_image)
        
        mirrored_image = mirror_image(image)
        cv2.imwrite(os.path.join(augmented_folder, f"{student_name}_mirrored_{i}.png"), mirrored_image)
        
        rotated_image = rotate_image(image, random.randint(0, 360))
        cv2.imwrite(os.path.join(augmented_folder, f"{student_name}_rotated_{i}.png"), rotated_image)

    # Setelah selesai augmentasi, pindahkan gambar-gambar hasil augmentasi ke folder dataset
    for file_name in os.listdir(augmented_folder):
        shutil.move(os.path.join(augmented_folder, file_name), os.path.join(dataset_folder, file_name))

    # Hapus folder augmented_dataset jika sudah kosong
    try:
        os.rmdir(augmented_folder)
    except OSError:
        pass

# Ubah path folder dataset sesuai kebutuhan Anda
if __name__ == "__main__":
    # Contoh pemanggilan fungsi augment_dataset
    student_name = "contoh_nama_mahasiswa"
    augment_dataset(student_name)