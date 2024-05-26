import cv2
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import create_dataset

# Mendapatkan path absolut dari direktori tempat skrip ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

# Fungsi untuk menampilkan frame dari webcam di dalam jendela Tkinter
def show_frame():
    ret, frame = cap.read()
    if ret:
        # Konversi frame OpenCV menjadi gambar PIL
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(frame)

        # Tampilkan frame di label
        label_webcam.imgtk = frame
        label_webcam.configure(image=frame)
    
    # Perbarui tampilan setiap 10 milidetik
    label_webcam.after(10, show_frame)

# Fungsi untuk menyimpan gambar dan menutup program saat sudah 2 gambar
def ambil_gambar(nama_mahasiswa):
    global count
    # Definisikan path folder dataset2 di sini (gunakan path absolut)
    folder_path = os.path.join(SCRIPT_DIR, "dataset2", nama_mahasiswa)  
    os.makedirs(folder_path, exist_ok=True)  # pastikan folder terbuat jika belum ada
    
    image_path = os.path.join(folder_path, f"{nama_mahasiswa}_{count}.png")
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(image_path, frame)
        count += 1
        if count == 1:
            messagebox.showinfo("Info", "Mohon lepas atribut atau ganti gaya.")
        elif count == 2:
            messagebox.showinfo("Info", "Gambar telah disimpan di folder dataset2.")

            # Panggil fungsi augmented_images dari create_dataset.py
            create_dataset.augmented_images(input_folder=f"dataset2/{nama_mahasiswa}",
                                             output_folder=folder_path,
                                             num_augmented_images=10)
            
            cap.release()
            cv2.destroyAllWindows()
            root.destroy()

# Fungsi yang akan dipanggil saat tombol Ambil Gambar ditekan
def on_ambil_gambar_clicked():
    nama_mahasiswa = name_entry.get()
    ambil_gambar(nama_mahasiswa)

# Buat GUI Tkinter
root = tk.Tk()
root.title("Input Nama Mahasiswa")

# Label untuk tampilan webcam
label_webcam = tk.Label(root)
label_webcam.pack(side="left", padx=10, pady=10)

# Label dan Entry untuk input nama mahasiswa
name_label = tk.Label(root, text="Nama Mahasiswa:")
name_label.pack()
name_entry = tk.Entry(root)
name_entry.pack(padx=10, pady=10)

# Tombol Ambil Gambar
ambil_button = tk.Button(root, text="Ambil Gambar", command=on_ambil_gambar_clicked)
ambil_button.pack()

# Inisialisasi count untuk menghitung jumlah gambar yang telah diambil
count = 0

# Tampilkan frame webcam
show_frame()

# Jalankan loop Tkinter
root.mainloop()