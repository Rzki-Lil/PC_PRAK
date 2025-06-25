import os
import zipfile
import shutil
from werkzeug.utils import secure_filename

def extract_dataset_zip(zip_file, extract_to='dataset'):
    try:
        # Buat direktori ekstraksi jika belum ada
        if os.path.exists(extract_to):
            shutil.rmtree(extract_to)
        os.makedirs(extract_to, exist_ok=True)
        
        # Ekstrak file ZIP
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Dapatkan semua nama file dalam ZIP
            all_files = zip_ref.namelist()
            
            # Cari folder train secara rekursif - abaikan folder test
            train_files = []
            
            for file_path in all_files:
                # Periksa apakah file ini ada dalam folder train
                if '/train/' in file_path and not file_path.endswith('/'):
                    if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                        train_files.append(file_path)
            
            # Jika tidak ada folder train yang ditemukan, coba pola alternatif
            if not train_files:
                for file_path in all_files:
                    # Cari folder yang berakhiran 'train'
                    path_parts = file_path.split('/')
                    for i, part in enumerate(path_parts):
                        if part.lower() == 'train' and i < len(path_parts) - 1:
                            if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                                train_files.append(file_path)
            
            if not train_files:
                return False, "Tidak ditemukan folder 'train' dalam ZIP. Pastikan ada folder 'train' yang berisi subfolder 'normal' dan 'cataract'"
            
            # Ekstrak file train dan atur ulang struktur
            train_normal_files = []
            train_cataract_files = []
            
            for file_path in train_files:
                if '/normal/' in file_path.lower():
                    # Ekstrak ke dataset/train/normal/
                    filename = os.path.basename(file_path)
                    target_path = os.path.join(extract_to, 'train', 'normal', filename)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    with zip_ref.open(file_path) as source, open(target_path, 'wb') as target:
                        target.write(source.read())
                    train_normal_files.append(filename)
                    
                elif '/cataract/' in file_path.lower():
                    # Ekstrak ke dataset/train/cataract/
                    filename = os.path.basename(file_path)
                    target_path = os.path.join(extract_to, 'train', 'cataract', filename)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    with zip_ref.open(file_path) as source, open(target_path, 'wb') as target:
                        target.write(source.read())
                    train_cataract_files.append(filename)
        
        # Validasi struktur yang diekstrak
        if len(train_normal_files) == 0:
            return False, "Tidak ada gambar ditemukan dalam folder 'train/normal'"
        
        if len(train_cataract_files) == 0:
            return False, "Tidak ada gambar ditemukan dalam folder 'train/cataract'"
        
        # Siapkan pesan sukses - hanya tampilkan data train
        message = f"Dataset berhasil diekstrak! Train - Normal: {len(train_normal_files)}, Cataract: {len(train_cataract_files)}"
        
        return True, message
        
    except zipfile.BadZipFile:
        return False, "File ZIP tidak valid atau rusak"
    except Exception as e:
        return False, f"Error saat mengekstrak ZIP: {str(e)}"

def validate_zip_file(file):
    """Validasi file ZIP yang diupload"""
    if not file:
        return False, "Tidak ada file yang diupload"
    
    if file.filename == '':
        return False, "Tidak ada file yang dipilih"
    
    if not file.filename.lower().endswith('.zip'):
        return False, "File harus berformat ZIP"
    
    # Hapus semua batasan ukuran file
    return True, file

def get_dataset_info():
    """Dapatkan informasi tentang dataset saat ini"""
    dataset_path = 'dataset'
    train_path = os.path.join(dataset_path, 'train')
    
    if not os.path.exists(train_path):
        return None
    
    train_normal_path = os.path.join(train_path, 'normal')
    train_cataract_path = os.path.join(train_path, 'cataract')
    
    info = {
        'exists': True,
        'normal_count': 0,
        'cataract_count': 0
    }
    
    # Hitung gambar train saja
    if os.path.exists(train_normal_path):
        info['normal_count'] = len([f for f in os.listdir(train_normal_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    if os.path.exists(train_cataract_path):
        info['cataract_count'] = len([f for f in os.listdir(train_cataract_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    return info
