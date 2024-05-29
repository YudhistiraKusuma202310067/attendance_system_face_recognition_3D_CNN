import tkinter as tk
import subprocess
import cv2
import PIL.Image, PIL.ImageTk
from tkinter import messagebox

def call_register():
    root.withdraw()  # Menyembunyikan jendela aplikasi
    release_webcam()  # Melepaskan sumber daya webcam
    register_process = subprocess.Popen(["python", "register2.py"])
    register_process.wait()  # Menunggu proses register selesai
    root.deiconify()  # Menampilkan kembali jendela aplikasi setelah proses register selesai
    show_webcam()  # Memulai kembali tampilan webcam setelah proses register selesai

def call_training():
    messagebox.showinfo("Training", "Proses training dimulai, mohon tunggu hingga proses training selesai")
    subprocess.Popen(["python", "training_fixed.py"])

def call_attendance():
    root.withdraw()  # Menyembunyikan jendela aplikasi
    release_webcam()  # Melepaskan sumber daya webcam
    register_process = subprocess.Popen(["python", "attendance.py"])
    register_process.wait()  # Menunggu proses attendance selesai
    root.deiconify()  # Menampilkan kembali jendela aplikasi setelah proses attendance selesai
    show_webcam()  # Memulai kembali tampilan webcam setelah proses attendance selesai

def release_webcam():
    cap.release()
    cv2.destroyAllWindows()

def show_webcam():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(cv2image)
        imgtk = PIL.ImageTk.PhotoImage(image=img)

        # Tampilkan webcam view di sebelah kiri
        label_webcam.imgtk = imgtk
        label_webcam.configure(image=imgtk)

        # Tampilkan tombol-tombol di bawahnya
        register_button.pack(side=tk.TOP, padx=10, pady=10)
        training_button.pack(side=tk.TOP, padx=10, pady=10)
        attendance_button.pack(side=tk.TOP, padx=10, pady=10)

        root.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

root = tk.Tk()
root.title("IBI Kesatuan Attendance System")

# Buat label untuk menampilkan webcam view
label_webcam = tk.Label(root)
label_webcam.pack(side=tk.LEFT, padx=10, pady=10)

# Buat tombol-tombol untuk register, training, dan attendance
register_button = tk.Button(root, text="Register", command=call_register)
training_button = tk.Button(root, text="Training", command=call_training)
attendance_button = tk.Button(root, text="Attendance", command=call_attendance)

# Mulai menampilkan webcam view
show_webcam()

root.mainloop()

# import tkinter as tk
# from PIL import Image, ImageTk
# import subprocess

# def call_register():
#     root.withdraw()  # Menyembunyikan jendela aplikasi
#     register_process = subprocess.Popen(["python", "register2.py"])
#     register_process.wait()  # Menunggu proses register selesai
#     root.deiconify()  # Menampilkan kembali jendela aplikasi setelah proses register selesai

# def call_training():
#     subprocess.Popen(["python", "post_processing.py"])

# def call_attendance():
#     subprocess.Popen(["python", "attendance.py"])

# root = tk.Tk()
# root.title("IBI Kesatuan Attendance System")

# # Baca gambar
# logo_image = Image.open("LOGO_IBIK.png")
# logo_image = logo_image.resize((200, 200))  # Hapus argumen ANTIALIAS

# # Konversi gambar ke format yang dapat ditampilkan oleh Tkinter
# tk_image = ImageTk.PhotoImage(logo_image)

# # Buat label untuk menampilkan gambar
# label_logo = tk.Label(root, image=tk_image)
# label_logo.pack(side=tk.LEFT)

# # Buat tombol-tombol untuk register, training, dan attendance
# register_button = tk.Button(root, text="Register", command=call_register)
# training_button = tk.Button(root, text="Training", command=call_training)
# attendance_button = tk.Button(root, text="Attendance", command=call_attendance)

# # Tampilkan tombol-tombol di bawah gambar
# register_button.pack(side=tk.TOP, padx=10, pady=10)
# training_button.pack(side=tk.TOP, padx=10, pady=10)
# attendance_button.pack(side=tk.TOP, padx=10, pady=10)

# root.mainloop()

# import tkinter as tk
# import subprocess

# def call_register():
#     subprocess.Popen(["python", "register.py"])

# def call_training():
#     subprocess.Popen(["python", "post_processing.py"])

# def call_attendance():
#     subprocess.Popen(["python", "attendance.py"])

# root = tk.Tk()
# root.title("Aplikasi Manajemen Data")

# register_button = tk.Button(root, text="Register", command=call_register)
# training_button = tk.Button(root, text="Training", command=call_training)
# attendance_button = tk.Button(root, text="Attendance", command=call_attendance)

# register_button.pack()
# training_button.pack()
# attendance_button.pack()

# root.mainloop()
