import cv2
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from mtcnn import MTCNN
from keras.models import load_model
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

from PIL import Image, ImageTk

from Silent_Face_Anti_Spoofing_master.test import test

# Load trained model
model = load_model('model/test dataset8.h5')
dataset_dir = 'be_augmented_dataset_5'
labels = [folder.replace('_', ' ') for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]

# Create Tkinter window
root = Tk()
root.title("Evaluation")

# Load MTCNN for face detection
detector = MTCNN()

# Access webcam
cap = cv2.VideoCapture(0)

# Create a frame to hold the webcam feed
frame_label = Label(root)
frame_label.pack(side="top", padx=10)  # Place frame on the top side

# Create a label to display the detected face
# face_label = Label(root)
# face_label.pack(side="top", padx=10)  # Place face label on the top side

# Define global variable to control frame update
update_frame_running = True
y_pred = []
y_test = []

# Fungsi untuk membuka jendela pop-up ketika status pada tabel ditekan
def open_status_window():
    global y_test
    # Fungsi untuk menyimpan perubahan status
    def save_status():
        selected_status = new_status_var.get()
        if selected_status == "Michael Mervin Ruswan":
            status_array = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif selected_status == "Muhammad Ilham":
            status_array = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        elif selected_status == "Raihan Dwi Pratama":
            status_array = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        elif selected_status == "Raihan Dwi Win Cahya":
            status_array = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        elif selected_status == "Raiyana Jan Winata":
            status_array = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        elif selected_status == "Yudhistira Kusuma":
            status_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        else:
            status_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Default value
        
        y_test.append(status_array)

        # Tutup jendela pop-up
        status_window.destroy()
        plt.close()  # Tutup jendela plot matplotlib

    # Buat jendela pop-up baru
    status_window = Toplevel(root)
    status_window.title("True Label")
    
    # Atur lebar jendela
    status_window_width = 300
    status_window_height = 160
    screen_width = status_window.winfo_screenwidth()
    screen_height = status_window.winfo_screenheight()
    x_coordinate = (screen_width / 2) - (status_window_width / 2)
    y_coordinate = (screen_height / 2) - (status_window_height / 2)
    status_window.geometry("%dx%d+%d+%d" % (status_window_width, status_window_height, x_coordinate, y_coordinate))

    # Buat StringVar untuk menyimpan status baru
    new_status_var = StringVar()

    # Buat radio button untuk setiap opsi status
    radio_button_mervin = ttk.Radiobutton(status_window, text="Michael Mervin Ruswan", variable=new_status_var, value="Michael Mervin Ruswan", command=save_status)
    radio_button_ilham = ttk.Radiobutton(status_window, text="Muhammad Ilham", variable=new_status_var, value="Muhammad Ilham", command=save_status)
    radio_button_dp = ttk.Radiobutton(status_window, text="Raihan Dwi Pratama", variable=new_status_var, value="Raihan Dwi Pratama", command=save_status)
    radio_button_dw = ttk.Radiobutton(status_window, text="Raihan Dwi Win Cahya", variable=new_status_var, value="Raihan Dwi Win Cahya", command=save_status)
    radio_button_rai = ttk.Radiobutton(status_window, text="Raiyana Jan Winata", variable=new_status_var, value="Raiyana Jan Winata", command=save_status)
    radio_button_yudhis = ttk.Radiobutton(status_window, text="Yudhistira Kusuma", variable=new_status_var, value="Yudhistira Kusuma", command=save_status)

    # Susun radio button dan tombol "Save" dalam jendela pop-up
    radio_button_mervin.grid(row=0, column=0, padx=10, sticky="w")
    radio_button_ilham.grid(row=1, column=0, padx=10, sticky="w")
    radio_button_dp.grid(row=2, column=0, padx=10, sticky="w")
    radio_button_dw.grid(row=3, column=0, padx=10, sticky="w")
    radio_button_rai.grid(row=4, column=0, padx=10, sticky="w")
    radio_button_yudhis.grid(row=5, column=0, padx=10, sticky="w")

# Fungsi untuk menampilkan frame dari webcam di dalam jendela Tkinter
def show_frame():
    ret, frame = cap.read()
    if ret:
        # Konversi frame OpenCV menjadi gambar PIL
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(frame)

        # Tampilkan frame di label
        frame_label.imgtk = frame
        frame_label.configure(image=frame)
    
    # Perbarui tampilan setiap 10 milidetik jika update_frame_running True
    if update_frame_running:
        frame_label.after(10, show_frame)

# Function to stop updating frame
def stop_update_frame():
    global update_frame_running
    update_frame_running = False
   
# Function to update the webcam feed
def update_frame():
    ret, frame = cap.read()
    if ret:
        global y_pred, y_test

        # Detect faces using MTCNN
        faces = detector.detect_faces(frame)

        if faces:
            for face in faces:
                #Anti Spoof
                x, y, w, h = face['box']
                top_limit = max(0, y - 75)
                bottom_limit = min(frame.shape[0], y + h + 75)
                left_limit = max(0, x - 75)
                right_limit = min(frame.shape[1], x + w + 75)
                    
                # Crop area dari rambut hingga leher dengan area yang diperluas
                face_img_spoof = frame[top_limit:bottom_limit, left_limit:right_limit]

                # plt.imshow(face_img_spoof)
                # plt.axis('off')  # Turn off axis
                # plt.show()

                label = test(image=face_img_spoof,
                     model_dir="Silent_Face_Anti_Spoofing_master/resources/anti_spoof_models",
                     device_id=0)

                #Predict
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                x, y, w, h = face['box']
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (50, 50))
                face_img = face_img.reshape(1, 50, 50, 1)  # Reshape sesuai dengan dimensi model

                # Convert the face image to PIL format
                # face_img_2 = frame[y:y+h, x:x+w]
                # face_pil = Image.fromarray(face_img_2)
                # face_img_tk = ImageTk.PhotoImage(face_pil)
                # face_label.configure(image=face_img_tk)
                # face_label.image = face_img_tk

                # Recognize face
                if label == 1:
                    # Set y_test
                    open_status_window()

                    # Tampilkan wajah menggunakan matplotlib
                    face_img_2 = frame[y:y+h, x:x+w]
                    plt.imshow(face_img_2, cmap='gray')
                    plt.show()

                    result = model.predict(face_img)
                    y_pred.append(result)
                    idx = result.argmax(axis=1)
                    confidence = result.max(axis=1) * 100
                    predicted_as = "N/A"
                    for i in idx:
                        if confidence > 80:
                            predicted_as = "%s (%.2f %%)" % (labels[i], confidence)
                                # update_status_and_timestamp(labels[i])
                        else:
                            predicted_as = "N/A"

                        print("Predicted as :", labels[i], "Persentase:", confidence)

                else:
                    predicted_as = "Fake Face Detected!!!"

                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

                # Display label and confidence
                cv2.putText(frame, predicted_as, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            y_pred_array = np.array([arr[0] for arr in y_pred])
            y_test_array = np.array(y_test)

            # print("y_test", y_test)
            # print("y_test", y_test_array.argmax(axis=1))
            # print("y_pred", y_pred_array.argmax(axis=1))

            print(classification_report(y_test_array.argmax(axis=1),
                                        y_pred_array.argmax(axis=1)
                                        # ,target_names=predicted_labels
                                        ))
                        
        else:
            print("No face detected")

        # Convert the frame to a format that Tkinter can display
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        frame_label.imgtk = imgtk
        frame_label.configure(image=imgtk)

# Function to start evaluation
def start_evaluate():
    stop_update_frame()  # Stop updating frame
    frame_label.after(100, update_frame)  # Update frame after a slight delay

def cancel_action():
    global update_frame_running, y_test, y_pred
    update_frame_running = True
    
    y_test = []
    y_pred = []

    show_frame()

# Tampilkan frame webcam
show_frame()

# Create "Start Evaluate" button
evaluate_button = Button(root, text="Start Evaluate", command=start_evaluate, bg="green", fg="white", width=20)
evaluate_button.pack(side="top", padx=10, pady=5)

# Create "Cancel" button
cancel_button = Button(root, text="Cancel", command=cancel_action, bg="red", fg="white", width=20)
cancel_button.pack(side="top", padx=10, pady=5)

# Run the Tkinter event loop
root.mainloop()