# import cv2
# import os
# import Augmentor
# import matplotlib.pyplot as plt
# from mtcnn import MTCNN

# def augmented_images(input_folder, output_folder, num_augmented_images):
#     # Membuat objek Augmentor
#     p = Augmentor.Pipeline(input_folder, output_folder)

#     # Menambahkan operasi augmentasi ke pipeline
#     p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
#     p.flip_left_right(probability=0.5)
#     p.flip_top_bottom(probability=0.5)
#     p.crop_random(probability=0.5, percentage_area=0.8)
#     p.zoom_random(probability=0.5, percentage_area=0.8)
#     p.flip_random(probability=0.5)

#     # Eksekusi augmentasi untuk menghasilkan num_augmented_images gambar tambahan untuk setiap gambar input
#     p.sample(num_augmented_images)

# def detect_and_segment_faces(image_path, output_folder, num_augmented_images):
#     img = cv2.imread(image_path)

#     # Deteksi wajah menggunakan MTCNN
#     detector = MTCNN()
#     faces = detector.detect_faces(img)

#     for face in faces:
#         x, y, w, h = face['box']
#         face_image = img[y:y+h, x:x+w]
#         face_image = cv2.resize(face_image, (224, 224))

#         # Simpan gambar wajah dalam output_folder
#         face_img_segmentation = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
#         cv2.imwrite(os.path.join(output_folder, 'original_face.jpg'), face_img_segmentation)

#         # Augmentasi gambar wajah
#         augmented_images(output_folder, output_folder, num_augmented_images)

#         # Gambarkan kotak di sekitar wajah pada gambar asli
#         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

#     # Tampilkan gambar asli dengan kotak di sekitar wajah menggunakan matplotlib
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()

# # Path folder
# training_folder = 'be_augmented_dataset/Rai'
# output_folder = 'dataset_training_output'

# # Pastikan folder output ada
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # Lakukan segmentasi dan augmentasi untuk setiap gambar di folder training
# for image_file in os.listdir(training_folder):
#     image_path = os.path.join(training_folder, image_file)
#     detect_and_segment_faces(image_path, output_folder, num_augmented_images=200)

# # # Lakukan segmentasi dan augmentasi untuk gambar 1.jpg di folder training
# # image_path = 'dataset_training/7.png'
# # detect_and_segment_faces(image_path, output_folder, num_augmented_images=200)

# import cv2
# import os
# import Augmentor
# import matplotlib.pyplot as plt
# from mtcnn import MTCNN
# import numpy as np

# def augmented_images(input_folder, output_folder, num_augmented_images):
#     # Mendapatkan daftar gambar dari input_folder
#     images = os.listdir(input_folder)

#     for image_file in images:
#         image_path = os.path.join(input_folder, image_file)
#         img = cv2.imread(image_path)

#         # Deteksi wajah menggunakan MTCNN
#         detector = MTCNN()
#         faces = detector.detect_faces(img)

#         for idx, face in enumerate(faces):
#             x, y, w, h = face['box']
#             face_image = img[y:y+h, x:x+w]
#             face_image = cv2.resize(face_image, (224, 224))

#             # Simpan gambar wajah dalam output_folder
#             cv2.imwrite(os.path.join(output_folder, f'{image_file[:-4]}_face_{idx}.jpg'), face_image)

#             # Augmentasi gambar wajah
#             augmented_images = img_augmentation(face_image, num_augmented_images)

#             # Simpan gambar-gambar hasil augmentasi
#             for i, augmented_image in enumerate(augmented_images):
#                 cv2.imwrite(os.path.join(output_folder, f'{image_file[:-4]}_face_{idx}_augmented_{i}.jpg'), augmented_image)

# def img_augmentation(img, num_augmented_images):
#     h, w = img.shape[:2]  # Mendapatkan dimensi gambar dengan benar
#     center = (w // 2, h // 2)
#     M_rot_5 = cv2.getRotationMatrix2D(center, 5, 1.0)
#     M_rot_neg_5 = cv2.getRotationMatrix2D(center, -5, 1.0)
#     M_rot_10 = cv2.getRotationMatrix2D(center, 10, 1.0)
#     M_rot_neg_10 = cv2.getRotationMatrix2D(center, -10, 1.0)
#     M_trans_3 = np.float32([[1, 0, 3], [0, 1, 0]])
#     M_trans_neg_3 = np.float32([[1, 0, -3], [0, 1, 0]])
#     M_trans_6 = np.float32([[1, 0, 6], [0, 1, 0]])
#     M_trans_neg_6 = np.float32([[1, 0, -6], [0, 1, 0]])
#     M_trans_y3 = np.float32([[1, 0, 0], [0, 1, 3]])
#     M_trans_neg_y3 = np.float32([[1, 0, 0], [0, 1, -3]])
#     M_trans_y6 = np.float32([[1, 0, 0], [0, 1, 6]])
#     M_trans_neg_y6 = np.float32([[1, 0, 0], [0, 1, -6]])

#     imgs = []
#     imgs.append(cv2.warpAffine(img, M_rot_5, (w, h), borderValue=(255,255,255)))
#     imgs.append(cv2.warpAffine(img, M_rot_neg_5, (w, h), borderValue=(255,255,255)))
#     imgs.append(cv2.warpAffine(img, M_rot_10, (w, h), borderValue=(255,255,255)))
#     imgs.append(cv2.warpAffine(img, M_rot_neg_10, (w, h), borderValue=(255,255,255)))
#     imgs.append(cv2.warpAffine(img, M_trans_3, (w, h), borderValue=(255,255,255)))
#     imgs.append(cv2.warpAffine(img, M_trans_neg_3, (w, h), borderValue=(255,255,255)))
#     imgs.append(cv2.warpAffine(img, M_trans_6, (w, h), borderValue=(255,255,255)))
#     imgs.append(cv2.warpAffine(img, M_trans_neg_6, (w, h), borderValue=(255,255,255)))
#     imgs.append(cv2.warpAffine(img, M_trans_y3, (w, h), borderValue=(255,255,255)))
#     imgs.append(cv2.warpAffine(img, M_trans_neg_y3, (w, h), borderValue=(255,255,255)))
#     imgs.append(cv2.warpAffine(img, M_trans_y6, (w, h), borderValue=(255,255,255)))
#     imgs.append(cv2.warpAffine(img, M_trans_neg_y6, (w, h), borderValue=(255,255,255)))
#     imgs.append(cv2.add(img, 10))
#     imgs.append(cv2.add(img, 30))
#     imgs.append(cv2.add(img, -10))
#     imgs.append(cv2.add(img, -30))
#     imgs.append(cv2.add(img, 15))
#     imgs.append(cv2.add(img, 45))
#     imgs.append(cv2.add(img, -15))
#     imgs.append(cv2.add(img, -45))

#     return imgs

# # Path folder
# training_folder = 'dataset_testing_augmented/ilham'
# output_folder = 'dataset_training_output'

# # Pastikan folder output ada
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # Lakukan augmentasi untuk setiap gambar di folder training
# augmented_images(training_folder, output_folder, num_augmented_images=200)

import cv2
import os
import numpy as np

def augmented_images(input_folder, output_folder, num_augmented_images):
    # Mendapatkan daftar gambar dari input_folder
    images = os.listdir(input_folder)

    for image_file in images:
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path)

        # Augmentasi gambar wajah
        augmented_faces = img_augmentation(img, num_augmented_images)

        # Simpan gambar-gambar hasil augmentasi
        for idx, augmented_face in enumerate(augmented_faces):
            cv2.imwrite(os.path.join(output_folder, f'{image_file[:-4]}_augmented_{idx}.jpg'), augmented_face)

def img_augmentation(img, num_augmented_images):
    h, w = img.shape[:2]  # Mendapatkan dimensi gambar dengan benar

    # Augmentasi dengan flipping
    flip_horizontal = cv2.flip(img, 1)
    # flip_vertical = cv2.flip(img, 0)
    # flip_both = cv2.flip(img, -1)

    # Augmentasi dengan scaling
    scale_up = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
    scale_down = cv2.resize(img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR)

    # Augmentasi dengan cropping
    crop_center = img[h//4:3*h//4, w//4:3*w//4]

    # Augmentasi dengan mirroring
    mirror_horizontal = img[:, ::-1]
    # mirror_vertical = img[::-1, :]

    # Augmentasi dengan rotasi
    M_rot_15 = cv2.getRotationMatrix2D((w/2, h/2), 15, 1.0)
    M_rot_neg_15 = cv2.getRotationMatrix2D((w/2, h/2), -15, 1.0)
    rotate_15 = cv2.warpAffine(img, M_rot_15, (w, h))
    rotate_neg_15 = cv2.warpAffine(img, M_rot_neg_15, (w, h))

    augmented_imgs = [
        flip_horizontal, 
        # flip_vertical, flip_both,
        scale_up, scale_down,
        crop_center,
        mirror_horizontal, 
        # mirror_vertical,
        rotate_15, rotate_neg_15
    ]

    return augmented_imgs * num_augmented_images

# # Path folder
# training_folder = 'dataset_testing_augmented/dw'
# output_folder = 'dataset_training_output'

# # Pastikan folder output ada
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # Lakukan augmentasi untuk setiap gambar di folder training
# augmented_images(training_folder, output_folder, num_augmented_images=10)  # Ubah num_augmented_images menjadi 21