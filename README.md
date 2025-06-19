# 🔬 Sistem Deteksi Katarak

Sistem deteksi katarak menggunakan machine learning dengan ekstraksi fitur GLCM (Gray-Level Co-occurrence Matrix) dan algoritma SVM (Support Vector Machine).

## 📋 Deskripsi

Aplikasi web ini dapat:
- Melatih model SVM menggunakan dataset gambar mata
- Mendeteksi katarak dari gambar mata yang diunggah
- Menampilkan fitur tekstur GLCM dan tingkat kepercayaan prediksi
- Memberikan penjelasan detail tentang fitur-fitur yang digunakan

## 🛠️ Teknologi yang Digunakan

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn, scikit-image
- **Computer Vision**: OpenCV
- **Frontend**: HTML, CSS, JavaScript
- **Styling**: CSS dengan gradient dan animasi

## 📁 Struktur Dataset
Download dataset terlebih dahulu : https://www.kaggle.com/datasets/jr2ngb/cataractdataset
Pastikan struktur dataset Anda seperti berikut:
```
dataset/
├── 1_normal/          # Gambar mata normal
│   ├── normal1.jpg
│   ├── normal2.jpg
│   └── ...
└── 2_cataract/        # Gambar mata katarak
    ├── cataract1.jpg
    ├── cataract2.jpg
    └── ...
```

## 🚀 Instalasi dan Menjalankan Program

### 1. Persyaratan Sistem
- Python 3.7 atau lebih baru
- pip (Python package installer)

### 2. Clone atau Download Repository
```bash
# Jika menggunakan git
git clone <repository-url>
cd PC_PRAK

# Atau extract file zip ke folder PC_PRAK
```

### 3. Buat dan Aktifkan Virtual Environment (Sangat Disarankan)

**Windows:**
```bash
# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
venv\Scripts\activate
```

**Linux/macOS:**
```bash
# Buat virtual environment
python3 -m venv venv

# Aktifkan virtual environment
source venv/bin/activate
```

> **💡 Tip**: Virtual environment akan mengisolasi dependensi proyek ini dari sistem Python global Anda, mencegah konflik antar package.

### 4. Install Dependencies
```bash
# Pastikan virtual environment sudah aktif (terlihat (venv) di command prompt)
pip install -r requirements.txt
```

### 5. Persiapkan Dataset
- Buat folder `dataset` di root directory
- Buat subfolder `1_normal` dan `2_cataract`
- Masukkan gambar mata normal ke folder `1_normal`
- Masukkan gambar mata katarak ke folder `2_cataract`
- Format gambar yang didukung: `.jpg`, `.jpeg`, `.png`

### 6. Jalankan Aplikasi
```bash
python app.py
```

### 7. Akses Aplikasi
Buka browser dan kunjungi: `http://localhost:5000`

## 📖 Cara Penggunaan

### 1. Melatih Model
1. Pastikan dataset sudah disiapkan dengan struktur yang benar
2. Klik tombol "🚀 Latih Model"
3. Tunggu proses training selesai
4. Model akan disimpan sebagai [`model.pkl`](model.pkl)

### 2. Prediksi Gambar
1. Pastikan model sudah dilatih
2. Upload gambar mata dengan cara:
   - Drag & drop ke area upload
   - Klik area upload untuk browse file
   - Paste dari clipboard (Ctrl+V)
3. Klik tombol "🔮 Prediksi"
4. Lihat hasil prediksi beserta fitur GLCM

## 🔧 Fitur Utama

### Ekstraksi Fitur
- **Color Features**: Histogram RGB dan statistik warna
- **Texture Features**: GLCM features (Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM)
- **Statistical Features**: Mean, Standard Deviation, Skewness, Kurtosis

### Machine Learning
- **Algorithm**: Support Vector Machine (SVM) dengan RBF kernel
- **Preprocessing**: Standard Scaler untuk normalisasi fitur
- **Validation**: 5-fold Cross Validation
- **Metrics**: Accuracy, Precision, Recall, F1-Score

### User Interface
- **Responsive Design**: Mendukung desktop dan mobile
- **Interactive**: Drag & drop, paste dari clipboard
- **Visual**: Animasi mata, gradient background
- **Informative**: Penjelasan detail fitur GLCM

## 📊 Output Model

Sistem akan menampilkan:
- **Prediksi**: Normal atau Cataract
- **Confidence Score**: Tingkat kepercayaan dalam persentase
- **Probability Distribution**: Probabilitas untuk setiap kelas
- **GLCM Features**: 6 fitur tekstur dengan penjelasan
- **Model Performance**: Akurasi training, testing, dan cross-validation

## 🔍 Troubleshooting

### Error "Module not found"
```bash
# Pastikan virtual environment aktif dan install ulang dependencies
pip install -r requirements.txt
```

### Error "No images found in dataset folders"
- Periksa struktur folder dataset
- Pastikan ada gambar dengan format `.jpg`, `.jpeg`, atau `.png`

### Error "Model not found"
- Latih model terlebih dahulu dengan klik "🚀 Latih Model"
- Pastikan file [`model.pkl`](model.pkl) ter-generate

### Performance Issues
- Gunakan gambar dengan resolusi wajar (tidak terlalu besar)
- Pastikan dataset seimbang antara kelas normal dan katarak

## 📝 Dependencies

Lihat file [`requirements.txt`](requirements.txt) untuk daftar lengkap dependencies:
- Flask==2.3.3
- opencv-python==4.8.1.78
- scikit-image==0.21.0
- scikit-learn==1.3.0
- numpy==1.24.3

## 🚪 Keluar dari Virtual Environment

Setelah selesai menggunakan aplikasi:
```bash
deactivate
```

## 📧 Support

Jika mengalami masalah, pastikan:
1. Virtual environment sudah aktif
2. Semua dependencies terinstall dengan benar
3. Dataset sudah disiapkan dengan struktur yang benar
4. Python versi 3.7 atau lebih baru

---

**Selamat menggunakan Sistem Deteksi Katarak! 🔬✨**
