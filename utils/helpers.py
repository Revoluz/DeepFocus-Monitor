"""
utils/helpers.py - Fungsi Utilitas Matematika

File ini berisi fungsi-fungsi matematika dasar yang digunakan
oleh modul lain dalam sistem, terutama untuk perhitungan
jarak dan normalisasi nilai.

Fungsi:
- euclidean_distance: Menghitung jarak antara dua titik landmark
- clamp: Membatasi nilai dalam range tertentu
"""

import math


def euclidean_distance(p1, p2):
    """
    Menghitung jarak Euclidean antara dua titik 2D.

    Digunakan untuk menghitung jarak antara landmark wajah,
    terutama untuk perhitungan EAR (Eye Aspect Ratio) dan
    Lip Distance (deteksi menguap).

    Args:
        p1: Titik pertama dengan atribut .x dan .y (MediaPipe landmark)
        p2: Titik kedua dengan atribut .x dan .y (MediaPipe landmark)

    Returns:
        float: Jarak Euclidean antara dua titik

    Rumus:
        distance = √((x1 - x2)² + (y1 - y2)²)

    Contoh penggunaan:
        >>> p1 = landmark[33]  # Sudut dalam mata kiri
        >>> p2 = landmark[133] # Sudut luar mata kiri
        >>> dist = euclidean_distance(p1, p2)
    """
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def clamp(value, min_val, max_val):
    """
    Membatasi nilai agar berada dalam range [min_val, max_val].

    Berguna untuk mencegah nilai keluar dari batas yang ditentukan,
    misalnya saat normalisasi koordinat atau pembatasan sudut.

    Args:
        value: Nilai yang akan dibatasi
        min_val: Batas minimum (inclusive)
        max_val: Batas maksimum (inclusive)

    Returns:
        Nilai yang sudah dibatasi dalam range [min_val, max_val]

    Contoh penggunaan:
        >>> clamp(150, 0, 100)
        100
        >>> clamp(-10, 0, 100)
        0
        >>> clamp(50, 0, 100)
        50
    """
    return max(min_val, min(value, max_val))
