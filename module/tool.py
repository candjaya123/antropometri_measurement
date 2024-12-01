import numpy as np
from playsound import playsound

def Play_buzzer():
    try:
        playsound('D:\PENS\PROYEK AKHIR\CODING\\buzzer1.mp3')
    except Exception as e:
        print(f"Error playing buzzer sound: {e}")

def Angle(var1, var2, var3):
    if var1 == None or var2 == None or var3 == None:
        return 0
    # print(f'var1[{var1}] - var2[{var2}] -var3[{var3}]')
    A, B, C = np.array(var1), np.array(var2), np.array(var3)
    
    ab = np.dot(A - B, C - B)
    a = np.linalg.norm(A - B)
    b = np.linalg.norm(C - B)
    
    # Check if either 'a' or 'b' is zero to avoid division by zero
    if a == 0 or b == 0:
        return np.nan  # Or return an appropriate value (e.g., 0)
    
    theta = np.arccos(ab / (a * b))
    return np.abs(np.degrees(theta))

def Y_angle(b, c):
    if b is None or c is None:
        return 0

    # Extracting the y-coordinates
    B, C = np.array(b), np.array(c)
    ab = -B[1] * (C[1] - B[1])
    
    a = B[1]
    b = np.linalg.norm(C - B)
    if b == 0:
        return np.nan
    
    theta = np.arccos(ab / (a * b))
    return np.abs(np.degrees(theta))

def Distance(var1, var2):
    if var1 == None or var2 == None:
        return 0
    A, B = np.array(var1), np.array(var2)
    return np.linalg.norm(A - B)
