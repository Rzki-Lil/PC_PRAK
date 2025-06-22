from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import pickle
import base64

from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import time

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

svm_model = None          # Model SVM yang sudah dilatih
color_scaler = None       # Scaler untuk normalisasi fitur warna
texture_scaler = None     # Scaler untuk normalisasi fitur tekstur
pca_reducer = None        # PCA untuk reduksi dimensi fitur
best_params = None        # Parameter terbaik dari training

def extract_color_features(image):
    # Ambil channel hijau saja (indeks 1 dari RGB)
    green_channel = image[:,:,1]
    
    #histogram dengan 32 bins (reduksi dari 256 untuk efisiensi)
    hist_g = cv2.calcHist([green_channel], [0], None, [32], [0, 256]).flatten()
    
    #statistik dasar dari channel hijau
    g_mean, g_std = np.mean(green_channel), np.std(green_channel) 
    g_min, g_max = np.min(green_channel), np.max(green_channel)  
    
    #percentile untuk representasi distribusi yang lebih baik
    g_25 = np.percentile(green_channel, 25)  # Kuartil 1
    g_75 = np.percentile(green_channel, 75)  # Kuartil 3
    
    #gabungan semua statistik warna
    color_stats = np.array([g_mean, g_std, g_min, g_max, g_25, g_75])
    
    # Gabungkan histogram dan statistik menjadi satu vektor fitur
    return np.concatenate((hist_g, color_stats))

def extract_texture_features(image):
    # Parameter GLCM yang dioptimasi untuk efisiensi
    distances = [1, 2]           # Jarak pixel yang dianalisis
    angles = [0, np.pi/4]       
    levels = 64                 
    
    scaled_image = (image / 256.0 * levels).astype(np.uint8)
    scaled_image = np.clip(scaled_image, 0, levels - 1)
    
    glcm = graycomatrix(scaled_image, distances, angles, levels=levels, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')        # Kontras: variasi intensitas
    homogeneity = graycoprops(glcm, 'homogeneity')  # Homogenitas: keseragaman tekstur
    energy = graycoprops(glcm, 'energy')            # Energi: uniformitas pola
    
    mean_intensity = np.mean(image)  
    std_intensity = np.std(image)   
    
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
    gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Peningkatan kontras menggunakan CLAHE (Contrast Limited Adaptive Histogram Equalization)
   
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    gray_img = clahe.apply(gray_img)
        
    # Terapkan CLAHE pada green channel untuk citra RGB
    rgb_img[:,:,1] = clahe.apply(rgb_img[:,:,1])
    
    return rgb_img, gray_img

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
                    rgb_img, gray_img = preprocess_image(aug_img)
                    color_features = extract_color_features(rgb_img)
                    texture_features = extract_texture_features(gray_img)
                        
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
                    rgb_img, gray_img = preprocess_image(aug_img)
                    color_features = extract_color_features(rgb_img)
                    texture_features = extract_texture_features(gray_img)
                       
                    all_color_features.append(color_features)
                    all_texture_features.append(texture_features)
                    labels.append(1)  # Label 1 untuk Cataract

    
    return np.array(all_color_features), np.array(all_texture_features), np.array(labels)

def train_model():

    global svm_model, color_scaler, texture_scaler, pca_reducer, best_params

    dataset_path = 'dataset'
    start_time = time.time()

    try:
        all_color_features, all_texture_features, labels = load_dataset_from_folders(dataset_path)
    except Exception as e:
        return False, f"Error loading dataset: {str(e)}"
    
    # Validasi dataset
    if len(all_color_features) == 0:
        return False, "No images found in dataset folders. Please check your dataset structure: dataset/train/{normal,cataract}"
    
    print(f"Total gambar dimuat (dengan augmentasi): {len(labels)}")
    print(f"Normal: {np.sum(labels==0)}, Katarak: {np.sum(labels==1)}")
    print(f"Bentuk fitur warna: {all_color_features.shape}")
    print(f"Bentuk fitur tekstur: {all_texture_features.shape}")
    
    # Normalisasi fitur menggunakan StandardScaler
    color_scaler = StandardScaler()
    texture_scaler = StandardScaler()
    
    scaled_color_features = color_scaler.fit_transform(all_color_features)
    scaled_texture_features = texture_scaler.fit_transform(all_texture_features)
    combined_features = np.concatenate((scaled_color_features, scaled_texture_features), axis=1)
    
    print(f"Bentuk fitur gabungan sebelum PCA: {combined_features.shape}")
    
    # Reduksi dimensi menggunakan PCA (mempertahankan 95% varians)
    pca_reducer = PCA(n_components=0.95, random_state=42)
    reduced_features = pca_reducer.fit_transform(combined_features)
    
    print(f"Bentuk fitur setelah PCA: {reduced_features.shape}")
    print(f"Rasio varians yang dijelaskan PCA: {pca_reducer.explained_variance_ratio_.sum():.3f}")
    
    # Grid parameter untuk berbagai kernel SVM
    param_grids = {
        'poly': {
            'kernel': ['poly'],
            'degree': [2], 
            'C': [0.01, 0.1, 1, 5], 
            'gamma': ['scale', 'auto'],
            'class_weight': ['balanced'],
            'coef0': [0, 1]
        },
        'rbf': {
            'kernel': ['rbf'],
            'C': [0.01, 0.1, 1],  
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'class_weight': ['balanced']
        },
        'linear': {
            'kernel': ['linear'],
            'C': [0.01, 0.1, 1],  
            'class_weight': ['balanced']
        }
    }
    
    best_overall_score = 0
    best_overall_model = None
    best_overall_params = None
    results_summary = []

    # Stratified K-Fold untuk validasi silang
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Uji berbagai rasio pembagian data train-test
    test_sizes = [0.2, 0.3, 0.4] # 80:20, 70:30, 60:40
    split_ratios = ['80:20', '70:30', '60:40']
    
    # Loop untuk setiap rasio pembagian data
    for test_size, split_ratio in zip(test_sizes, split_ratios):
        print(f"\nMenguji pembagian train-test {split_ratio}...")
        
        # Pembagian data train-test dengan stratifikasi
        X_train, X_test, y_train, y_test = train_test_split(
            reduced_features, labels, 
            test_size=test_size, 
            random_state=42, 
            stratify=labels
        )
        
        split_results = {
            'split_ratio': split_ratio,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'kernels': {}
        }

        # Loop untuk setiap kernel SVM
        for kernel_name, param_grid in param_grids.items():
            print(f"  Menguji kernel {kernel_name.upper()}...")

            # Inisialisasi SVM dengan probabilitas
            svm = SVC(probability=True, random_state=42)
            
            # Grid Search untuk mencari parameter terbaik
            grid_search = GridSearchCV(
                svm, param_grid, 
                cv=skf,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            try:
                # Latih model dengan grid search
                grid_search.fit(X_train, y_train)
                
                # Ambil model terbaik untuk kernel ini
                best_model = grid_search.best_estimator_
                
                # Evaluasi pada test set
                y_pred = best_model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=skf, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Deteksi overfitting
                train_accuracy = best_model.score(X_train, y_train)
                overfitting_gap = train_accuracy - test_accuracy
                
                # Hitung stabilitas validasi
                validation_stability = 1 - cv_std
                
                kernel_result = {
                    'best_params': grid_search.best_params_,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'overfitting_gap': overfitting_gap,
                    'validation_stability': validation_stability,
                    'model': best_model
                }
                
                split_results['kernels'][kernel_name] = kernel_result
                
                print(f"    Skor CV: {cv_mean:.4f} (±{cv_std:.4f})")
                print(f"    Akurasi Test: {test_accuracy:.4f}")
                print(f"    Gap Overfitting: {overfitting_gap:.4f}")
                
                # Composite score untuk ranking model (prioritas generalisasi)
                composite_score = (test_accuracy * 0.4 + 
                                 cv_mean * 0.3 + 
                                 validation_stability * 0.2 - 
                                 overfitting_gap * 0.1)
                
                if composite_score > best_overall_score:
                    best_overall_score = composite_score
                    best_overall_model = best_model
                    best_overall_params = {
                        'kernel': kernel_name,
                        'split_ratio': split_ratio,
                        'params': grid_search.best_params_,
                        'metrics': kernel_result,
                        'composite_score': composite_score
                    }
                
            except Exception as e:
                print(f"    Error dengan kernel {kernel_name}: {str(e)}")
                continue
        
        results_summary.append(split_results)
    
    # Validasi bahwa ada model yang berhasil dilatih
    if best_overall_model is None:
        return False, "No valid model could be trained. Please check your dataset."
    
    svm_model = best_overall_model
    best_params = best_overall_params
    
    #evaluasi final dengan rasio split terbaik
    best_test_size = test_sizes[split_ratios.index(best_params['split_ratio'])]
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
        reduced_features, labels, 
        test_size=best_test_size, 
        random_state=42, 
        stratify=labels
    )
    
    # Prediksi dan laporan final
    y_pred_final = svm_model.predict(X_test_final)
   
    # Simpan model dan semua komponen preprocessing
    model_data = {
        'model': svm_model,
        'color_scaler': color_scaler,
        'texture_scaler': texture_scaler,
        'pca_reducer': pca_reducer,
        'best_params': best_params,
        'training_time': time.time() - start_time,
        'results_summary': results_summary,
        'feature_dimensions': {
            'original': combined_features.shape[1],
            'after_pca': reduced_features.shape[1],
            'reduction_ratio': 1 - (reduced_features.shape[1] / combined_features.shape[1])
        }
    }
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    results_html = generate_training_results_html(results_summary, best_params,  time.time() - start_time, model_data['feature_dimensions'])
    
    return True, results_html

def generate_training_results_html(results_summary, best_params, training_time, feature_dims):

    best_info = best_params['metrics']
    
    html = f"""
<div class="best-model-summary">
    <h4>🏆 Model Terbaik Dipilih</h4>
    <div class="info-grid">
        <div class="info-item">
            <span class="label">Kernel:</span>
            <span class="value">{best_params['kernel'].upper()}</span>
        </div>
        <div class="info-item">
            <span class="label">Split Ratio:</span>
            <span class="value">{best_params['split_ratio']}</span>
        </div>
        <div class="info-item">
            <span class="label">Train Accuracy:</span>
            <span class="value">{best_info['train_accuracy']:.1%}</span>
        </div>
        <div class="info-item">
            <span class="label">Test Accuracy:</span>
            <span class="value">{best_info['test_accuracy']:.1%}</span>
        </div>
        <div class="info-item">
            <span class="label">CV Score:</span>
            <span class="value">{best_info['cv_mean']:.1%} (±{best_info['cv_std']:.1%})</span>
        </div>
        <div class="info-item">
            <span class="label">Overfitting Gap:</span>
            <span class="value {'good' if best_info['overfitting_gap'] < 0.05 else 'warning' if best_info['overfitting_gap'] < 0.1 else 'bad'}">{best_info['overfitting_gap']:.1%}</span>
        </div>
        <div class="info-item">
            <span class="label">Validation Stability:</span>
            <span class="value">{best_info['validation_stability']:.1%}</span>
        </div>
        <div class="info-item">
            <span class="label">Composite Score:</span>
            <span class="value">{best_params['composite_score']:.3f}</span>
        </div>
        <div class="info-item">
            <span class="label">Training Time:</span>
            <span class="value">{training_time:.1f}s</span>
        </div>
    </div>
</div>

<div class="best-model-summary">
    <h5>📊 Reduksi Dimensi Fitur</h5>
    <div class="info-grid">
        <div class="info-item">
            <span class="label">Dimensi Asli:</span>
            <span class="value">{feature_dims['original']}</span>
        </div>
        <div class="info-item">
            <span class="label">Setelah PCA:</span>
            <span class="value">{feature_dims['after_pca']}</span>
        </div>
        <div class="info-item">
            <span class="label">Reduksi:</span>
            <span class="value">{feature_dims['reduction_ratio']:.1%}</span>
        </div>
    </div>
</div>

<div class="comparison-table-container">
    <h4>📋 Perbandingan Split Ratio dan Kernel</h4>
    <div class="table-wrapper">
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Split Ratio</th>
                    <th>Train Size</th>
                    <th>Test Size</th>
                    <th>Kernel</th>
                    <th>CV Score</th>
                    <th>Test Accuracy</th>
                    <th>Overfitting Gap</th>
                    <th>Composite Score</th>
                </tr>
            </thead>
            <tbody>
"""
    
    all_scores = []
    for split_result in results_summary:
        for kernel_name, kernel_result in split_result['kernels'].items():
            composite_score = (kernel_result['test_accuracy'] * 0.4 + 
                             kernel_result['cv_mean'] * 0.3 + 
                             kernel_result['validation_stability'] * 0.2 - 
                             kernel_result['overfitting_gap'] * 0.1)
            all_scores.append(composite_score)
    
    best_composite_score = max(all_scores) if all_scores else 0
    
    # Generate table rows
    for split_result in results_summary:
        split_ratio = split_result['split_ratio']
        train_size = split_result['train_size']
        test_size = split_result['test_size']
        
        for kernel_name, kernel_result in split_result['kernels'].items():
            composite_score = (kernel_result['test_accuracy'] * 0.4 + 
                             kernel_result['cv_mean'] * 0.3 + 
                             kernel_result['validation_stability'] * 0.2 - 
                             kernel_result['overfitting_gap'] * 0.1)
            
            is_best = abs(composite_score - best_composite_score) < 1e-6
            row_class = "best-row" if is_best else ""
            
            gap_class = "good" if kernel_result['overfitting_gap'] < 0.05 else "warning" if kernel_result['overfitting_gap'] < 0.1 else "bad"
            
            html += f"""
                <tr class="{row_class}">
                    <td><strong>{split_ratio}</strong></td>
                    <td>{train_size}</td>
                    <td>{test_size}</td>
                    <td><strong>{kernel_name.upper()}</strong></td>
                    <td>{kernel_result['cv_mean']:.1%} ±{kernel_result['cv_std']:.1%}</td>
                    <td>{kernel_result['test_accuracy']:.1%}</td>
                    <td class="{gap_class}">{kernel_result['overfitting_gap']:.1%}</td>
                    <td><strong>{composite_score:.3f}</strong>{'🏆' if is_best else ''}</td>
                </tr>
            """
    
    html += """
            </tbody>
        </table>
    </div>
</div>
"""
    
    return html

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

def extract_glcm_info(gray_img):

    levels = 64
    scaled_gray = (gray_img / 256.0 * levels).astype(np.uint8)
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

def predict_image_core(img, include_glcm_info=False):
    # Load model components jika belum dimuat
    success, error_msg = load_model_components()
    if not success:
        return None, error_msg
    
    try:
        # Preprocessing gambar
        rgb_img, gray_img = preprocess_image(img)

        # Ekstraksi fitur warna dan tekstur
        color_features = extract_color_features(rgb_img)
        texture_features = extract_texture_features(gray_img)

        # Ekstraksi GLCM info jika diperlukan
        glcm_info = extract_glcm_info(gray_img) if include_glcm_info else None
        
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


def validate_uploaded_file(file):

    if 'file' not in file:
        return False, 'No file uploaded'
    
    uploaded_file = file['file']
    if uploaded_file.filename == '':
        return False, 'No file selected'
    
    if not uploaded_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return False, 'Invalid file type'
    
    return True, uploaded_file


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    success, message = train_model()
    return jsonify({'success': success, 'message': message})

@app.route('/predict', methods=['POST'])
def predict():
    is_valid, result = validate_uploaded_file(request.files)
    if not is_valid:
        return jsonify({'success': False, 'message': result})
    
    file = result
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        prediction_data, error = predict_image_core(img, include_glcm_info=True)
        
        if prediction_data:
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return jsonify({
                'success': True, 
                'prediction': prediction_data['result'], 
                'confidence': f"{prediction_data['confidence']:.1f}%",
                'probabilities': prediction_data['probabilities'], 
                'image_url': f"data:image/jpeg;base64,{img_base64}",
                'glcm_features': prediction_data.get('glcm_features', {}),
            })
        else:
            return jsonify({'success': False, 'message': error})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
