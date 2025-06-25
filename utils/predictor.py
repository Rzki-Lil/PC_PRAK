import numpy as np
import pickle
from .image_processing import preprocess_image, extract_color_features, extract_texture_features, extract_glcm_info

svm_model = None
color_scaler = None
texture_scaler = None
pca_reducer = None

def load_model_components():
    global svm_model, color_scaler, texture_scaler, pca_reducer
    
    if svm_model is None:
        try:
            with open('model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                svm_model = model_data['model']
                color_scaler = model_data['color_scaler']
                texture_scaler = model_data['texture_scaler']
                pca_reducer = model_data.get('pca_reducer')  
            return True, None
        except Exception as e:
            return False, f"Model not found or corrupted: {str(e)}. Please train the model first."
    return True, None

def predict_image_core(img, include_glcm_info=False):
    # Load model components jika belum dimuat
    success, error_msg = load_model_components()
    if not success:
        return None, error_msg
    
    try:
        # Preprocessing gambar - hanya RGB
        rgb_img = preprocess_image(img)

        # Ekstraksi fitur warna dan tekstur (keduanya menggunakan green channel)
        color_features = extract_color_features(rgb_img)
        texture_features = extract_texture_features(rgb_img)

        # Ekstraksi GLCM info jika diperlukan (gunakan RGB untuk akses green channel)
        glcm_info = extract_glcm_info(rgb_img) if include_glcm_info else None
        
        # Normalisasi fitur menggunakan scaler yang sudah dilatih
        scaled_color_features = color_scaler.transform([color_features])
        scaled_texture_features = texture_scaler.transform([texture_features])
        
        # Gabungkan fitur
        feature_vector = np.concatenate((scaled_color_features, scaled_texture_features), axis=1)
        
        # Terapkan PCA jika tersedia
        if pca_reducer is not None:
            feature_vector = pca_reducer.transform(feature_vector)
        
        # Prediksi menggunakan model SVM
        prediction = svm_model.predict(feature_vector)[0]
        probabilities = svm_model.predict_proba(feature_vector)[0]
        
        class_names = ['Normal', 'Cataract']
        result = class_names[prediction]
        confidence = probabilities[prediction] * 100
        
        if confidence < 80.0:
       
            return None, f"Tingkat kepercayaan terlalu rendah: {confidence}. Pastikan gambar yang Anda upload adalah gambar mata yang jelas dan berkualitas baik."
        
        print(f"Prediksi: {prediction}")
        print(f"Probabilitas: Normal={probabilities[0]:.4f}, Katarak={probabilities[1]:.4f}")
        print(f"Hasil: {result}")
        print(f"Kepercayaan: {confidence:.2f}%")
        
        # Prepare return data
        prediction_data = {
            'result': result,
            'confidence': confidence,
            'probabilities': {
                'normal': float(probabilities[0]),
                'cataract': float(probabilities[1])
            }
        }
        
        if include_glcm_info:
            prediction_data['glcm_features'] = glcm_info
        
        return prediction_data, None
        
    except Exception as e:
        return None, f"Error during prediction: {str(e)}"
