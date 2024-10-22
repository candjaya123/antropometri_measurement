import cv2
import mediapipe as mp
import numpy as np
import os

# Inisialisasi Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Folder yang berisi gambar
folder_path = './Pengukuran'
output_folder = './Hasil_Deteksi'

# Fungsi untuk menghitung jarak pixel antara dua landmark
def calculate_pixel_distance(landmark1, landmark2, image_width, image_height):
    x1, y1 = int(landmark1[0] * image_width), int(landmark1[1] * image_height)
    x2, y2 = int(landmark2[0] * image_width), int(landmark2[1] * image_height)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Fungsi untuk mengonversi dari pixel ke cm menggunakan hasil regresi
def convert_pixel_to_cm(pixel_value, a=0.002, b=0.5, c=1.0):
    return a * (pixel_value ** 2) + b * pixel_value + c

# Fungsi untuk memproses gambar dan menghitung pengukuran
def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        image_height, image_width, _ = image.shape
        
        # Hitung tinggi badan (rata-rata dari NOSE ke LEFT_HEEL dan NOSE ke RIGHT_HEEL)
        height_left = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.NOSE.value],
                                               landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value],
                                               image_width, image_height)
        height_right = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.NOSE.value],
                                                landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value],
                                                image_width, image_height)
        height_pixel = (height_left + height_right) / 2
        height_cm = convert_pixel_to_cm(height_pixel,0.0048,7.2197,2909)

        # Hitung lebar tangan (rata-rata dari LEFT_SHOULDER ke LEFT_WRIST dan RIGHT_SHOULDER ke RIGHT_WRIST)
        left_hand_span = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                                  image_width, image_height)
        right_hand_span = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                                   image_width, image_height)
        hand_span_pixel = (left_hand_span + right_hand_span) / 2
        hand_span_cm = convert_pixel_to_cm(hand_span_pixel, )

        # Hitung lebar bahu (dari bahu kiri ke bahu kanan)
        shoulder_width_pixel = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], 
                                                        image_width, image_height)
        shoulder_width_cm = convert_pixel_to_cm(shoulder_width_pixel)

        # Hitung panjang paha (rata-rata paha kiri dan paha kanan)
        left_thigh_length = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                     image_width, image_height)
        right_thigh_length = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                                      image_width, image_height)
        thigh_length_pixel = (left_thigh_length + right_thigh_length) / 2
        thigh_length_cm = convert_pixel_to_cm(thigh_length_pixel)

        # Tampilkan hasil pengukuran pada gambar
        cv2.putText(image, f'Height: {height_cm:.2f} cm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Hand Span: {hand_span_cm:.2f} cm', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Shoulder Width: {shoulder_width_cm:.2f} cm', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Thigh Length: {thigh_length_cm:.2f} cm', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return image
    else:
        return None

# Buat folder output jika belum ada
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop melalui semua file gambar di folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        # Dapatkan path file gambar
        image_path = os.path.join(folder_path, filename)
        
        # Proses gambar dan ambil pengukuran
        output_image = process_image(image_path)
        
        if output_image is not None:
            # Simpan gambar dengan hasil deteksi
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, output_image)
        else:
            print(f"Pose tidak terdeteksi pada gambar: {filename}")

print(f"Pengukuran selesai. Hasil disimpan di folder {output_folder}")