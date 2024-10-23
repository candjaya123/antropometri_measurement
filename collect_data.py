import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# Inisialisasi Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Folder yang berisi gambar
folder_path = './Raw_data'

# Fungsi untuk menghitung jarak pixel antara dua landmark
def calculate_pixel_distance(landmark1, landmark2, image_width, image_height):
    x1, y1 = int(landmark1[0] * image_width), int(landmark1[1] * image_height)
    x2, y2 = int(landmark2[0] * image_width), int(landmark2[1] * image_height)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Fungsi untuk memproses gambar dan menghitung pengukuran
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
        height = (height_left + height_right) / 2

        # Hitung lebar tangan (rata-rata dari LEFT_SHOULDER ke LEFT_WRIST dan RIGHT_SHOULDER ke RIGHT_WRIST)
        left_hand_span = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                                  image_width, image_height)
        right_hand_span = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                                   image_width, image_height)
        hand_span = (left_hand_span + right_hand_span) / 2

        # Hitung lebar bahu (dari bahu kiri ke bahu kanan)
        shoulder_width = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], 
                                                  image_width, image_height)

        # Hitung panjang paha (rata-rata paha kiri dan paha kanan)
        left_thigh_length = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                     image_width, image_height)
        right_thigh_length = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                                      image_width, image_height)
        thigh_length = (left_thigh_length + right_thigh_length) / 2

        # Hitung panjang kaki dari hip ke ankle (rata-rata kaki kiri dan kaki kanan)
        left_leg_length = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                   landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                                   image_width, image_height)
        right_leg_length = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                                    image_width, image_height)
        leg_length = (left_leg_length + right_leg_length) / 2

        return height, hand_span, shoulder_width, thigh_length, leg_length
    else:
        return None, None, None, None, None

# File CSV untuk menyimpan hasil pengukuran
csv_file = 'hasil_pengukuran.csv'

# Buka file CSV untuk menulis data dengan pemisah ';' dan desimal ','
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    # Tulis header kolom
    writer.writerow(['Nama', 'Tinggi Badan (pixel)', 'Lebar Tangan (pixel)', 'Lebar Bahu (pixel)', 'Panjang Paha (pixel)', 'Panjang Kaki (pixel)'])
    
    # Loop melalui semua file gambar di folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            # Dapatkan path file gambar
            image_path = os.path.join(folder_path, filename)
            
            # Ekstrak nama dari file (tanpa ekstensi) dan ubah ke lowercase
            name = os.path.splitext(filename)[0].lower()
            
            # Proses gambar dan ambil pengukuran
            height, hand_span, shoulder_width, thigh_length, leg_length = process_image(image_path)
            
            if height is not None:
                # Konversi hasil ke format desimal dengan koma
                height_str = f"{height:.2f}".replace('.', ',')
                hand_span_str = f"{hand_span:.2f}".replace('.', ',')
                shoulder_width_str = f"{shoulder_width:.2f}".replace('.', ',')
                thigh_length_str = f"{thigh_length:.2f}".replace('.', ',')
                leg_length_str = f"{leg_length:.2f}".replace('.', ',')
    
                # Tulis hasil ke file CSV
                writer.writerow([name, height_str, hand_span_str, shoulder_width_str, thigh_length_str, leg_length_str])
            else:
                print(f"Pose tidak terdeteksi pada gambar: {filename}")


print(f"Pengukuran selesai. Hasil disimpan di {csv_file}")
