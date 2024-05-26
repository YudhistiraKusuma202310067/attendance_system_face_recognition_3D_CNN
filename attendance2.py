# import cv2
# from tkinter import *
# from tkinter import ttk
# from PIL import Image, ImageTk
# from mtcnn import MTCNN
# from keras.models import load_model
# import numpy as np
# import os
# import json
# from datetime import datetime
# from Silent_Face_Anti_Spoofing_master.test import test

# # Load trained model
# model = load_model('model/test dataset8.h5')
# dataset_dir = 'be_augmented_dataset_5'
# labels = [folder.replace('_', ' ') for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]

# # Load MTCNN for face detection
# detector = MTCNN()

# # Function to end attendance and close the application
# def end_attendance():
#     cap.release()
#     cv2.destroyAllWindows()
#     root.destroy()

# # Function to update student table based on selected class
# def update_students(event=None):
#     selected_class = class_var.get()
#     if selected_class == "Select Class":
#         clear_table()
#     else:
#         students = class_data[selected_class]["student"]
#         clear_table()
#         for student in students:
#             student_data = (student["npm"], student["name"], "Alpha", "-")
#             student_table.insert("", "end", values=student_data)

# # Function to clear student table
# def clear_table():
#     student_table.delete(*student_table.get_children())

# # Function to update status and timestamp based on face recognition
# def update_status_and_timestamp(student_name):
#     for row_id in student_table.get_children():
#         student = student_table.item(row_id)["values"]
#         if student[1] == student_name:
#             if student[3] == "-":  # Check if timestamp is empty
#                 student_table.item(row_id, values=(student[0], student[1], "Presence", datetime.now().strftime("%d %b %Y %H:%M:%S")))
#             break

# # Create Tkinter window
# root = Tk()
# root.title("Attendance")

# # Access webcam
# cap = cv2.VideoCapture(0)

# # Load class data from JSON
# with open("student_class.json", "r") as f:
#     class_data = json.load(f)

# # Create a frame to hold the webcam feed
# frame_label = Label(root)
# frame_label.pack(side="top", padx=10, pady=10)  # Place frame on the top side

# # Create dropdown for class selection
# class_var = StringVar()
# class_var.set("Select Class")  # Set default value to placeholder
# class_dropdown = ttk.Combobox(root, textvariable=class_var, state="readonly")
# class_dropdown["values"] = ["Select Class"] + list(class_data.keys())
# class_dropdown.pack(side="top", anchor="e", padx=10, pady=10)  # Place dropdown on top
# class_dropdown.bind("<<ComboboxSelected>>", update_students)  # Bind function to event

# # Create student table
# student_table = ttk.Treeview(root, columns=("npm", "name", "status", "timestamp"), show="headings")
# student_table.heading("npm", text="NPM")
# student_table.heading("name", text="Name")
# student_table.heading("status", text="Status")
# student_table.heading("timestamp", text="Timestamp")
# student_table.pack(side="top", padx=10)  # Place table below dropdown

# # Set initial value for attendance_running
# attendance_running = False

# # Function to update the webcam feed
# def update_frame():
#     ret, frame = cap.read()
#     if ret:
#         # Detect faces using MTCNN
#         faces = detector.detect_faces(frame)

#         label = test(image=frame,
#                      model_dir="Silent_Face_Anti_Spoofing_master/resources/anti_spoof_models",
#                      device_id=0)

#         if label == 1:
#             for face in faces:
#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#                 x, y, w, h = face['box']
#                 face_img = gray[y:y+h, x:x+w]
#                 face_img = cv2.resize(face_img, (50, 50))
#                 face_img = face_img.reshape(1, 50, 50, 1)  # Reshape sesuai dengan dimensi model

#                 # Recognize face
#                 result = model.predict(face_img)
#                 idx = result.argmax(axis=1)
#                 confidence = result.max(axis=1) * 100
#                 predicted_as = "N/A"
#                 for i in idx:
#                     if confidence > 80:
#                         predicted_as = "%s (%.2f %%)" % (labels[i], confidence)
#                         update_status_and_timestamp(labels[i])
#                     else:
#                         predicted_as = "N/A"

#                     print("Predicted as :", labels[i], "Persentase:", confidence)

#                 # Draw rectangle around face
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

#                 # Display label and confidence
#                 cv2.putText(frame, predicted_as, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#             # Convert the frame to a format that Tkinter can display
#             img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             img = Image.fromarray(img)
#             imgtk = ImageTk.PhotoImage(image=img)
#             frame_label.imgtk = imgtk
#             frame_label.configure(image=imgtk)

#             # Call the update_frame function again after 10 milliseconds
#             frame_label.after(10, update_frame)
#         else:
#             # Process for fake face detected
#             for face in faces:
#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#                 x, y, w, h = face['box']
#                 face_img = gray[y:y+h, x:x+w]
#                 face_img = cv2.resize(face_img, (50, 50))
#                 face_img = face_img.reshape(1, 50, 50, 1)  # Reshape sesuai dengan dimensi model

#                 predicted_as = "Fake Face Detected!!!"

#                 # Draw rectangle around face
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

#                 # Display label and confidence
#                 cv2.putText(frame, predicted_as, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#             # Convert the frame to a format that Tkinter can display
#             img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             img = Image.fromarray(img)
#             imgtk = ImageTk.PhotoImage(image=img)
#             frame_label.imgtk = imgtk
#             frame_label.configure(image=imgtk)

#             # Call the update_frame function again after 10 milliseconds
#             frame_label.after(10, update_frame)

# # # Call the update_frame function to start capturing the webcam feed
# # update_frame()

# # # Create "Start Attendance" button
# # start_button = Button(root, text="Start Attendance", command=update_frame, bg="green", fg="white")
# # start_button.pack(side="top", padx=10, pady=10)

# # # Create "End Attendance" button
# # end_button = Button(root, text="End Attendance", command=end_attendance, bg="red", fg="white")
# # end_button.pack(side="top", padx=10, pady=10)

# # Function to start attendance
# def start_attendance():
#     global attendance_running
#     attendance_running = True
#     update_frame()
#     start_button.pack_forget()
#     end_button.pack(side="top", padx=10, pady=10)

# # Function to end attendance
# def end_attendance():
#     global attendance_running
#     attendance_running = False
#     start_button.pack(side="top", padx=10, pady=10)
#     end_button.pack_forget()
#     cap.release()
#     cv2.destroyAllWindows()
#     root.destroy()

# # Create "Start Attendance" button
# start_button = Button(root, text="Start Attendance", command=start_attendance, bg="green", fg="white")
# start_button.pack(side="top", padx=10, pady=10)

# # Create "End Attendance" button
# end_button = Button(root, text="End Attendance", command=end_attendance, bg="red", fg="white")

# # Run the Tkinter event loop
# root.mainloop()

import cv2
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from mtcnn import MTCNN
from keras.models import load_model
import numpy as np
import os
import json
from datetime import datetime
from Silent_Face_Anti_Spoofing_master.test import test

# Load trained model
model = load_model('model/test dataset8.h5')
dataset_dir = 'be_augmented_dataset_5'
labels = [folder.replace('_', ' ') for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]

# Load MTCNN for face detection
detector = MTCNN()

# Function to end attendance and close the application
def end_attendance():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

# Function to update student table based on selected class
def update_students(event=None):
    selected_class = class_var.get()
    if selected_class == "Select Class":
        clear_table()
    else:
        students = class_data[selected_class]["student"]
        clear_table()
        for student in students:
            student_data = (student["npm"], student["name"], "Alpha", "-")
            student_table.insert("", "end", values=student_data)

# Function to clear student table
def clear_table():
    student_table.delete(*student_table.get_children())

# Function to update status and timestamp based on face recognition
def update_status_and_timestamp(student_name):
    for row_id in student_table.get_children():
        student = student_table.item(row_id)["values"]
        if student[1] == student_name:
            if student[3] == "-":  # Check if timestamp is empty
                student_table.item(row_id, values=(student[0], student[1], "Presence", datetime.now().strftime("%d %b %Y %H:%M:%S")))
            break

# Create Tkinter window
root = Tk()
root.title("Attendance")

# Access webcam
cap = cv2.VideoCapture(0)

# Load class data from JSON
with open("student_class.json", "r") as f:
    class_data = json.load(f)

# Create a frame to hold the webcam feed
frame_label = Label(root)
frame_label.pack(side="top", padx=10, pady=10)  # Place frame on the top side

# Create dropdown for class selection
class_var = StringVar()
class_var.set("Select Class")  # Set default value to placeholder
class_dropdown = ttk.Combobox(root, textvariable=class_var, state="readonly")
class_dropdown["values"] = ["Select Class"] + list(class_data.keys())
class_dropdown.pack(side="top", anchor="e", padx=10, pady=10)  # Place dropdown on top
class_dropdown.bind("<<ComboboxSelected>>", update_students)  # Bind function to event

# Create student table
student_table = ttk.Treeview(root, columns=("npm", "name", "status", "timestamp"), show="headings")
student_table.heading("npm", text="NPM")
student_table.heading("name", text="Name")
student_table.heading("status", text="Status")
student_table.heading("timestamp", text="Timestamp")
student_table.pack(side="top", padx=10)  # Place table below dropdown

# Set initial value for attendance_running
attendance_running = False

# Function to update the webcam feed
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
                        update_status_and_timestamp(labels[i])
                    else:
                        predicted_as = "N/A"

                    print("Predicted as :", labels[i], "Persentase:", confidence)

                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

                # Display label and confidence
                cv2.putText(frame, predicted_as, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

                # Display label and confidence
                cv2.putText(frame, predicted_as, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Convert the frame to a format that Tkinter can display
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            frame_label.imgtk = imgtk
            frame_label.configure(image=imgtk)

            # Call the update_frame function again after 10 milliseconds
            frame_label.after(10, update_frame)

# # Call the update_frame function to start capturing the webcam feed
# update_frame()

# # Create "Start Attendance" button
# start_button = Button(root, text="Start Attendance", command=update_frame, bg="green", fg="white")
# start_button.pack(side="top", padx=10, pady=10)

# # Create "End Attendance" button
# end_button = Button(root, text="End Attendance", command=end_attendance, bg="red", fg="white")
# end_button.pack(side="top", padx=10, pady=10)

# Function to start attendance
def start_attendance():
    global start_date
    start_date = datetime.now().strftime("%A, %d %B %Y %H:%M:%S")
    global attendance_running
    attendance_running = True
    update_frame()
    start_button.pack_forget()
    end_button.pack(side="top", padx=10, pady=10)

# Function to end attendance
# def end_attendance():
#     global attendance_running
#     attendance_running = False
#     start_button.pack(side="top", padx=10, pady=10)
#     end_button.pack_forget()
#     cap.release()
#     cv2.destroyAllWindows()
#     root.destroy()

def end_attendance():
    global attendance_running
    attendance_running = False
    start_button.pack(side="top", padx=10, pady=10)
    end_button.pack_forget()
    cap.release()
    cv2.destroyAllWindows()

    # Create JSON data for attendance
    attendance_data = {}
    selected_class = class_var.get()
    if selected_class != "Select Class":
        end_date = datetime.now().strftime("%A, %d %B %Y %H:%M:%S")
        lecturer = class_data[selected_class]["lecturer"]

        students = class_data[selected_class]["student"]
        student_attendance = []
        for student in students:
            npm = student["npm"]
            name = student["name"]
            status = "Alpha"
            timestamp = "-"
            # Find the latest status and timestamp displayed in student_table
            for row_id in student_table.get_children():
                row_values = student_table.item(row_id)["values"]
                # Convert npm to string for comparison
                npm_str = str(npm)
                if str(row_values[0]) == npm_str:  # Check if the current row corresponds to the student
                    # Always update status and timestamp
                    status = row_values[2]
                    timestamp = row_values[3]
                    break  # Once found, no need to continue looping
            student_attendance.append({"npm": npm_str, "name": name, "Status": status, "Timestamp": timestamp})

        attendance_data[selected_class] = {
            "start_date": start_date,
            "end_date": end_date,
            "lecturer": lecturer,
            "student": student_attendance
        }

        # Save attendance data to JSON file
        with open("attendance.json", "w") as json_file:
            json.dump(attendance_data, json_file, indent=4)

    root.destroy()

# Create "Start Attendance" button
start_button = Button(root, text="Start Attendance", command=start_attendance, bg="green", fg="white")
start_button.pack(side="top", padx=10, pady=10)

# Create "End Attendance" button
end_button = Button(root, text="End Attendance", command=end_attendance, bg="red", fg="white")

# Run the Tkinter event loop
root.mainloop()