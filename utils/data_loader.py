import os
import cv2
import numpy as np
from .image_processing import augment_image, preprocess_image, extract_color_features, extract_texture_features

def load_dataset_from_folders(dataset_path):
    all_color_features = []
    all_texture_features = []
    labels = []
    
    train_path = os.path.join(dataset_path, 'train')
    
    if not os.path.exists(train_path):
        print("Folder train tidak ditemukan, menggunakan folder root dataset")
        train_path = dataset_path
    else:
        print("Menggunakan struktur folder train")
    
    normal_folder = os.path.join(train_path, 'normal')
    if os.path.exists(normal_folder):
        normal_files = [f for f in os.listdir(normal_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Memproses {len(normal_files)} gambar normal...")
        
        for file in normal_files:
            image_path = os.path.join(normal_folder, file)
            img = cv2.imread(image_path)
            if img is not None:
        
                augmented_images = augment_image(img)
                for aug_img in augmented_images:
                    rgb_img = preprocess_image(aug_img)  
                    color_features = extract_color_features(rgb_img)
                    texture_features = extract_texture_features(rgb_img)
                        
                    all_color_features.append(color_features)
                    all_texture_features.append(texture_features)
                    labels.append(0)  

    cataract_folder = os.path.join(train_path, 'cataract')
    if os.path.exists(cataract_folder):
        cataract_files = [f for f in os.listdir(cataract_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Memproses {len(cataract_files)} gambar katarak...")
        
        for file in cataract_files:
            image_path = os.path.join(cataract_folder, file)
            img = cv2.imread(image_path)
            if img is not None:

                augmented_images = augment_image(img)
                for aug_img in augmented_images:
                    rgb_img = preprocess_image(aug_img) 
                    color_features = extract_color_features(rgb_img)
                    texture_features = extract_texture_features(rgb_img)
                       
                    all_color_features.append(color_features)
                    all_texture_features.append(texture_features)
                    labels.append(1)  # Label 1 untuk Cataract

    
    return np.array(all_color_features), np.array(all_texture_features), np.array(labels)
