import cv2
from tkinter import *
from PIL import Image, ImageTk
from mtcnn import MTCNN
from keras.models import load_model
import numpy as np
import os
from Silent_Face_Anti_Spoofing_master.test import test

# Function to end attendance and close the application
def end_attendance():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

# Load trained model
model = load_model('model/test dataset8.h5')
dataset_dir = 'be_augmented_dataset_5'
labels = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]

# Load MTCNN for face detection
detector = MTCNN()

# Create Tkinter window
root = Tk()
root.title("Attendance")

# Create a frame to hold the webcam feed
frame_label = Label(root)
frame_label.pack(side="left")

# Access webcam
cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if ret:
        # Detect faces using MTCNN
        faces = detector.detect_faces(frame)

        label = test(image=frame,
                     model_dir="Silent_Face_Anti_Spoofing_master/resources/anti_spoof_models",
                     device_id=0)

        if label == 1:
            for face in faces:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                x, y, w, h = face['box']
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (50, 50))
                face_img = face_img.reshape(1, 50, 50, 1)  # Reshape sesuai dengan dimensi model

                # Recognize face
                result = model.predict(face_img)
                idx = result.argmax(axis=1)
                confidence = result.max(axis=1) * 100
                predicted_as = "N/A"
                for i in idx:
                    if confidence > 80:
                        predicted_as = "%s (%.2f %%)" % (labels[i], confidence)
                    else:
                        predicted_as = "N/A"

                    print("Predicted as :", labels[i], "Persentase:", confidence)

                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Display label and confidence
                cv2.putText(frame, predicted_as, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Convert the frame to a format that Tkinter can display
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            frame_label.imgtk = imgtk
            frame_label.configure(image=imgtk)

            # Call the update_frame function again after 10 milliseconds
            frame_label.after(10, update_frame)
        else:
            # Process for fake face detected
            for face in faces:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                x, y, w, h = face['box']
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (50, 50))
                face_img = face_img.reshape(1, 50, 50, 1)  # Reshape sesuai dengan dimensi model

                predicted_as = "Fake Face Detected!!!"

                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Display label and confidence
                cv2.putText(frame, predicted_as, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Convert the frame to a format that Tkinter can display
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            frame_label.imgtk = imgtk
            frame_label.configure(image=imgtk)

            # Call the update_frame function again after 10 milliseconds
            frame_label.after(10, update_frame)

# Call the update_frame function to start capturing the webcam feed
update_frame()

# Create "End Attendance" button
end_button = Button(root, text="End Attendance", command=end_attendance, bg="red", fg="white")
end_button.pack(anchor=NE)

# Run the Tkinter event loop
root.mainloop()