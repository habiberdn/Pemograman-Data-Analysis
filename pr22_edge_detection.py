import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

#Fungsi untuk membaca gambar menggunakan OpenCV
def read_image(file_path):
    image = cv2.imread(file_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Mengonversi gambar ke format RGB

#Fungsi untuk menerapkan deteksi tapi menggunakan algoritma Canny
def apply_canny_edge_detection(image, low_threshold, high_threshold):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges

#Fungsi untuk menerapkan deteksi tapi menggunakan filter sobel
def apply_sobel_edge_detection(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    sobel_edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return sobel_edges

#Fungsi untuk menerapkan deteksi tepi menggunakan filter laplacian
def apply_laplacian_edge_detection(image):
    laplacian_edges = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_edges = cv2.convertScaleAbs(laplacian_edges)
    return laplacian_edges

#Fungsi untuk menerapkan deteksi tepi menggunakan filter robert
def apply_roberts_edge_detection(image):
    kernelx = np.array([[1, 0], [0, -1]], dtype=int)
    kernely = np.array([[0, 1], [-1, 0]], dtype=int)
    img_x = cv2.filter2D(image, cv2.CV_16S, kernelx)
    img_y = cv2.filter2D(image, cv2.CV_16S, kernely)
    abs_img_x = cv2.convertScaleAbs(img_x)
    abs_img_y = cv2.convertScaleAbs(img_y)
    roberts_edge = cv2.addWeighted(abs_img_x, 0.5, abs_img_y, 0.5, 0)
    return roberts_edge

#Fungsi untuk menerapkan deteksi tepi menggunakan filter prewitt
def apply_prewitt_edge_detection(image):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    img_x = cv2.filter2D(image, cv2.CV_16S, kernelx)
    img_y = cv2.filter2D(image, cv2.CV_16S, kernely)
    abs_img_x = cv2.convertScaleAbs(img_x)
    abs_img_y = cv2.convertScaleAbs(img_y)
    prewitt_edge = cv2.addWeighted(abs_img_x, 0.5, abs_img_y, 0.5, 0)
    return prewitt_edge

#Fungsi untuk menerapkan deteksi tepi menggunakan Laplacian of Gaussian (LoG)
def apply_log_edge_detection(image):
    #Mengaburkan gambar dengan Gaussian
    blurred = cv2.GaussianBlur(image, (3,3), 0)
    #Menerapkan deteksi tepi Laplacian
    log_edges = cv2.Laplacian(blurred, cv2.CV_64F)
    log_edges = cv2.convertScaleAbs(log_edges)
    return log_edges

#Fungsi untuk menampilkan gambar-gambar dalam satu form menggunakan matplotlib
def display_images(original, edges_roberts, edges_sobel, edges_prewitt, edges_log, edges_canny):
    fig, axes = plt.subplots(2, 3, figsize=(20,10))

    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(edges_roberts, cmap='gray')
    axes[0, 1].set_title('Roberts Edge Detection')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(edges_sobel, cmap='gray')
    axes[0, 2].set_title('Sobel Edge Detection')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(edges_prewitt, cmap='gray')
    axes[1, 0].set_title('Prewitt Edge Detection')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(edges_log, cmap='gray')
    axes[1, 1].set_title('Laplacian of Gaussian (LoG) Edge Detection')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(edges_canny, cmap='gray')
    axes[1, 2].set_title('Canny Edge Detection')
    axes[1, 2].axis('off')

    plt.show()

#Fungsi untuk membuka dialog file dan memilih gambar
def choose_image_file():
    root = Tk()
    root.withdraw() #Menyembunyikan jendela utama Tkinter
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
    return file_path

#Main Script
if __name__ == "__main__":
    #memilih gambar dari komputer
    input_image_path = choose_image_file()
    if not input_image_path:
        print("No image selected.")
        exit()

    #Membaca gambar
    original_image = read_image(input_image_path)

    #Mengkonversi gambar ke grayscale sebelum penerapan deteksi tepi
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

    #Menerapkan deteksi tepi roberts
    edges_roberts = apply_roberts_edge_detection(gray_image)

    #Menerapkan deteksi tepi sobel
    edges_sobel = apply_sobel_edge_detection(gray_image)

    #Menerapkan deteksi tepi prewitt
    edges_prewitt = apply_prewitt_edge_detection(gray_image)

    #Menerapkan deteksi tepi Laplacian of Gaussian(LoG)
    edges_log = apply_log_edge_detection(gray_image)

    #Menerapkan deteksi tepi Canny
    edges_canny = apply_canny_edge_detection(gray_image, low_threshold=50, high_threshold=150)

    #Menampilkan gambar asli dan hasil deteksi tepi
    display_images(original_image, edges_roberts, edges_sobel, edges_prewitt, edges_log, edges_canny)

    print("Proses deteksi tepi selesai.")