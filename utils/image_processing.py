import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_color_features(image):
    # Ambil channel hijau saja (indeks 1 dari RGB)
    green_channel = image[:,:,1]
    
    #histogram dengan 32 bins (reduksi dari 256 untuk efisiensi)
    hist_g = cv2.calcHist([green_channel], [0], None, [32], [0, 256]).flatten()
    
    #statistik dasar dari channel hijau
    g_mean, g_std = np.mean(green_channel), np.std(green_channel) 
    g_min, g_max = np.min(green_channel), np.max(green_channel)  

    #gabungan semua statistik warna
    color_stats = np.array([g_mean, g_std, g_min, g_max])
    
    # Gabungkan histogram dan statistik menjadi satu vektor fitur
    return np.concatenate((hist_g, color_stats))

def extract_texture_features(image):
    # Parameter GLCM yang dioptimasi untuk efisiensi
    distances = [1, 2]           # Jarak pixel yang dianalisis
    angles = [0, np.pi/4]       
    levels = 64                 

    green_channel = image[:,:,1] if len(image.shape) == 3 else image
    scaled_image = (green_channel / 256.0 * levels).astype(np.uint8)
    scaled_image = np.clip(scaled_image, 0, levels - 1)
    
    glcm = graycomatrix(scaled_image, distances, angles, levels=levels, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')        # Kontras: variasi intensitas
    homogeneity = graycoprops(glcm, 'homogeneity')  # Homogenitas: keseragaman tekstur
    energy = graycoprops(glcm, 'energy')            # Energi: uniformitas pola
    
    mean_intensity = np.mean(green_channel)  
    std_intensity = np.std(green_channel)   
    
    texture_stats = np.array([mean_intensity, std_intensity])
    
    return np.concatenate((contrast.flatten(), homogeneity.flatten(), energy.flatten(), texture_stats))

def augment_image(image):
    augmented_images = []
    
    # Gambar asli
    augmented_images.append(image)
    
    #horizontal
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)

    # Rotasi kecil (-3° dan +3°)
    for angle in [-3, 3]:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        augmented_images.append(rotated)

    # Adjustment brightness (perubahan kecerahan)
    for beta in [-10, 10]:
        adjusted = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
        augmented_images.append(adjusted)
    
    return augmented_images

def preprocess_image(image):
    resized = cv2.resize(image, (300, 300))
    
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Terapkan CLAHE pada green channel untuk citra RGB
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    rgb_img[:,:,1] = clahe.apply(rgb_img[:,:,1])
    
    return rgb_img

def extract_glcm_info(rgb_img):
    levels = 64
    # Gunakan green channel dari RGB
    green_channel = rgb_img[:,:,1]
        
    scaled_gray = (green_channel / 256.0 * levels).astype(np.uint8)
    scaled_gray = np.clip(scaled_gray, 0, levels - 1)
    
    glcm = graycomatrix(scaled_gray, [1, 2], [0, np.pi/4], levels=levels, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')

    return {
        'contrast': float(np.mean(contrast)),
        'homogeneity': float(np.mean(homogeneity)),
        'energy': float(np.mean(energy))
    }
