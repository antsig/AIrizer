# Perbandingan Evolusi Sistem AI Document Intelligence

Berikut adalah rincian perbandingan antara prototipe sistem yang paling awal dengan sistem versi terbaru yang baru saja kita perbarui:

## 1. Kemampuan Mengelola Dokumen (Input Data)

| Fitur / Komponen | Sistem Versi Awal (Prototipe) | Sistem Versi Saat Ini (Advanced) |
| :--- | :--- | :--- |
| **Sumber Data Utama** | Mengandalkan *dataset* dummy (sintetik) hasil generasi acak dan *copy-paste* teks manual. | Mendukung pembacaan *real-world documents*. Pengguna dapat mengunggah file **PDF, Word (.docx), dan Teks (.txt)** secara langsung. |
| **Parsing Dokumen** | Tidak ada proses ekstraksi dokumen. | Terdapat `DocumentParser` berbasis `PyPDF2` dan `python-docx` yang mengekstrak isi teks dari file biner secara otomatis. |

## 2. Fitur Pencarian Semantik (Vector DB)

| Fitur / Komponen | Sistem Versi Awal (Prototipe) | Sistem Versi Saat Ini (Advanced) |
| :--- | :--- | :--- |
| **Batas Ruang Pencarian** | Hanya mencari dari 50 dokumen sintetik yang terdapat di dalam file `dataset.json`. | Bisa mencari ke dokumen *custom*. Pengguna cukup memasukkan path folder di komputer (Misal: `D:\Arsip_Laporan`), dan AI akan membacanya secara massal untuk di-*index* ke ChromaDB. |

## 3. Akurasi dan Antarmuka Evaluasi

| Fitur / Komponen | Sistem Versi Awal (Prototipe) | Sistem Versi Saat Ini (Advanced) |
| :--- | :--- | :--- |
| **Visualisasi UI** | Hanya menampilkan kategori akhir (Misal: "Surat Resmi") tanpa indikator tingkat keyakinan AI. | Menampilkan **Akurasi / Confidence Score** (Misal: "Akurasi: 98.5%") di layar Dashboard Streamlit. |
| **Pipeline Pemodelan** | Kosong. Direktori `notebooks/` belum terisi skrip evaluasi apapun. | Sudah dilengkapi skrip interaktif. Terdapat alat untuk mengukur **Accuracy, F1-Score**, dan menampilkan visualisasi **Confusion Matrix** menggunakan `scikit-learn` & `seaborn`. |
| **Pengembangan AI** | Sistem terkunci pada penggunakan model *Zero-Shot Classification* tanpa panduan peningkatan. | Menyediakan *template/blueprint* lengkap untuk melatih ulang (***Fine-Tuning***) model `IndoBERT` secara mandiri agar bisa lebih spesifik mengerti bahasa unik dari perusahaan Anda. |

## Ringkasan Eksekutif

> [!TIP]
> **Evolusi Terbesar:**
> Sistem awal berfungsi sekadar sebagai **Proof of Concept (PoC)** untuk membuktikan bahwa teknologi AI bisa memahami teks. Sedangkan sistem saat ini telah menjelma menjadi **Aplikasi Skala Praktik** yang benar-benar siap dipasang di kantor karena sudah bisa mengurai file PDF/Word sungguhan, menelusuri isi folder spesifik di komputer Anda, dan memiliki ruang kerja (notebooks) bagi seorang Data Scientist untuk terus menyempurnakan AI-nya.
