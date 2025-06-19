from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import pickle
import base64

from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 

svm_model = None
color_scaler = None
texture_scaler = None

def extract_color_features(image):
    hist_r = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([image], [2], None, [256], [0, 256]).flatten()

    r_mean, r_std = np.mean(image[:,:,0]), np.std(image[:,:,0])
    g_mean, g_std = np.mean(image[:,:,1]), np.std(image[:,:,1])
    b_mean, b_std = np.mean(image[:,:,2]), np.std(image[:,:,2])

    color_stats = np.array([r_mean, r_std, g_mean, g_std, b_mean, b_std])

    return np.concatenate((hist_r, hist_g, hist_b, color_stats))

def extract_texture_features(image):
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(image, distances, angles, levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    asm = graycoprops(glcm, 'ASM')
    

    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    skewness = np.mean(((image - mean_intensity) / std_intensity) ** 3)
    kurtosis = np.mean(((image - mean_intensity) / std_intensity) ** 4)
    
    texture_stats = np.array([mean_intensity, std_intensity, skewness, kurtosis])
    
    return np.concatenate((contrast.flatten(), dissimilarity.flatten(), homogeneity.flatten(),
                           energy.flatten(), correlation.flatten(), asm.flatten(), texture_stats))

def preprocess_image(image, enhance_contrast=True):
    resized = cv2.resize(image, (224, 224))
    
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_img = clahe.apply(gray_img)

        for i in range(3):
            rgb_img[:,:,i] = clahe.apply(rgb_img[:,:,i])
    
    return rgb_img, gray_img

def train_model():
    global svm_model, color_scaler, texture_scaler

    normal_folder = 'dataset/1_normal'
    cataract_folder = 'dataset/2_cataract'
    
    all_color_features = []
    all_texture_features = []
    labels = []
    
    print("Starting feature extraction...")
    
    # Process normal images
    if os.path.exists(normal_folder):
        normal_files = [f for f in os.listdir(normal_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Processing {len(normal_files)} normal images...")
        
        for file in normal_files:
            image_path = os.path.join(normal_folder, file)
            img = cv2.imread(image_path)
            if img is not None:
                rgb_img, gray_img = preprocess_image(img)

                color_features = extract_color_features(rgb_img)
                texture_features = extract_texture_features(gray_img)
                
                all_color_features.append(color_features)
                all_texture_features.append(texture_features)
                labels.append(0)  

    if os.path.exists(cataract_folder):
        cataract_files = [f for f in os.listdir(cataract_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Processing {len(cataract_files)} cataract images...")
        
        for file in cataract_files:
            image_path = os.path.join(cataract_folder, file)
            img = cv2.imread(image_path)
            if img is not None:
                rgb_img, gray_img = preprocess_image(img)
                
                color_features = extract_color_features(rgb_img)
                texture_features = extract_texture_features(gray_img)
                
                all_color_features.append(color_features)
                all_texture_features.append(texture_features)
                labels.append(1)  

    if len(all_color_features) == 0:
        return False, "No images found in dataset folders"
    
    # Convert to numpy arrays
    color_features = np.array(all_color_features)
    texture_features = np.array(all_texture_features)
    labels = np.array(labels)
    
    print(f"Color features shape: {color_features.shape}")
    print(f"Texture features shape: {texture_features.shape}")
    print(f"Label distribution: Normal={np.sum(labels==0)}, Cataract={np.sum(labels==1)}")

    color_scaler = StandardScaler()
    texture_scaler = StandardScaler()
    
    scaled_color_features = color_scaler.fit_transform(color_features)
    scaled_texture_features = texture_scaler.fit_transform(texture_features)

    combined_features = np.concatenate((scaled_color_features, scaled_texture_features), axis=1)
    
    print(f"Combined features shape: {combined_features.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        combined_features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"Train size: {X_train.shape}")
    print(f"Test size: {X_test.shape}")
    print(f"Train label distribution: Normal={np.sum(y_train==0)}, Cataract={np.sum(y_train==1)}")
    print(f"Test label distribution: Normal={np.sum(y_test==0)}, Cataract={np.sum(y_test==1)}")
   
    print("Training SVM model...")

    svm_model = SVC(
        kernel='rbf', 
        C=1.0, 
        gamma='scale', 
        probability=True, 
        class_weight='balanced',
        random_state=42
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"CV Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    svm_model.fit(X_train, y_train)
    
    train_accuracy = svm_model.score(X_train, y_train)
    test_accuracy = svm_model.score(X_test, y_test)
    
    y_pred = svm_model.predict(X_test)
    
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Cataract']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model_data = {
        'model': svm_model,
        'color_scaler': color_scaler,
        'texture_scaler': texture_scaler,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return True, f"""
<div class="metrics-grid">
    <div class="metric-item">
        <div class="metric-label">Akurasi Training</div>
        <div class="metric-value">{train_accuracy:.1%}</div>
    </div>
    <div class="metric-item">
        <div class="metric-label">Akurasi Testing</div>
        <div class="metric-value">{test_accuracy:.1%}</div>
    </div>
    <div class="metric-item">
        <div class="metric-label">Cross Validation</div>
        <div class="metric-value">{cv_scores.mean():.1%} (±{cv_scores.std() * 2:.1%})</div>
    </div>
</div>

<div class="classification-report">
    <h5>📊 Laporan Klasifikasi</h5>
    <table class="report-table">
        <thead>
            <tr>
                <th>Kelas</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Support</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Normal</strong></td>
                <td>{classification_report(y_test, y_pred, target_names=['Normal', 'Cataract'], output_dict=True)['Normal']['precision']:.2f}</td>
                <td>{classification_report(y_test, y_pred, target_names=['Normal', 'Cataract'], output_dict=True)['Normal']['recall']:.2f}</td>
                <td>{classification_report(y_test, y_pred, target_names=['Normal', 'Cataract'], output_dict=True)['Normal']['f1-score']:.2f}</td>
                <td>{classification_report(y_test, y_pred, target_names=['Normal', 'Cataract'], output_dict=True)['Normal']['support']:.0f}</td>
            </tr>
            <tr>
                <td><strong>Cataract</strong></td>
                <td>{classification_report(y_test, y_pred, target_names=['Normal', 'Cataract'], output_dict=True)['Cataract']['precision']:.2f}</td>
                <td>{classification_report(y_test, y_pred, target_names=['Normal', 'Cataract'], output_dict=True)['Cataract']['recall']:.2f}</td>
                <td>{classification_report(y_test, y_pred, target_names=['Normal', 'Cataract'], output_dict=True)['Cataract']['f1-score']:.2f}</td>
                <td>{classification_report(y_test, y_pred, target_names=['Normal', 'Cataract'], output_dict=True)['Cataract']['support']:.0f}</td>
            </tr>
            <tr style="background: rgba(255, 255, 255, 0.1);">
                <td><strong>Rata-rata</strong></td>
                <td>{classification_report(y_test, y_pred, target_names=['Normal', 'Cataract'], output_dict=True)['weighted avg']['precision']:.2f}</td>
                <td>{classification_report(y_test, y_pred, target_names=['Normal', 'Cataract'], output_dict=True)['weighted avg']['recall']:.2f}</td>
                <td>{classification_report(y_test, y_pred, target_names=['Normal', 'Cataract'], output_dict=True)['weighted avg']['f1-score']:.2f}</td>
                <td>{len(y_test)}</td>
            </tr>
        </tbody>
    </table>
</div>"""

def predict_image(image_path):
    global svm_model, color_scaler, texture_scaler
    
    if svm_model is None:
        try:
            with open('model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                svm_model = model_data['model']
                color_scaler = model_data['color_scaler']
                texture_scaler = model_data['texture_scaler']
        except Exception as e:
            return None, f"Model not found or corrupted: {str(e)}. Please train the model first."
    
    img = cv2.imread(image_path)
    if img is None:
        return None, "Could not read image"
    
    rgb_img, gray_img = preprocess_image(img)
    
    color_features = extract_color_features(rgb_img)
    texture_features = extract_texture_features(gray_img)
    
    try:
        scaled_color_features = color_scaler.transform([color_features])
        scaled_texture_features = texture_scaler.transform([texture_features])
        
        feature_vector = np.concatenate((scaled_color_features, scaled_texture_features), axis=1)
        
        prediction = svm_model.predict(feature_vector)[0]
        probabilities = svm_model.predict_proba(feature_vector)[0]
        
        class_names = ['Normal', 'Cataract']
        
        result = class_names[prediction]
        confidence = probabilities[prediction] * 100
        
        print(f"Prediction: {prediction}")
        print(f"Probabilities: Normal={probabilities[0]:.4f}, Cataract={probabilities[1]:.4f}")
        print(f"Result: {result}")
        print(f"Confidence: {confidence:.2f}%")
        
        return result, f"Confidence: {confidence:.1f}%"
        
    except Exception as e:
        return None, f"Error during prediction: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    success, message = train_model()
    return jsonify({'success': success, 'message': message})

def predict_image_from_array(img):
    """Predict image directly from numpy array without saving to disk"""
    global svm_model, color_scaler, texture_scaler
    
    if svm_model is None:
        try:
            with open('model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                svm_model = model_data['model']
                color_scaler = model_data['color_scaler']
                texture_scaler = model_data['texture_scaler']
        except Exception as e:
            return None, f"Model not found or corrupted: {str(e)}. Please train the model first."
    
    try:
        rgb_img, gray_img = preprocess_image(img)

        color_features = extract_color_features(rgb_img)
        texture_features = extract_texture_features(gray_img)

        glcm = graycomatrix(gray_img, [1, 2], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')
        dissimilarity = graycoprops(glcm, 'dissimilarity')
        homogeneity = graycoprops(glcm, 'homogeneity')
        energy = graycoprops(glcm, 'energy')
        correlation = graycoprops(glcm, 'correlation')
        asm = graycoprops(glcm, 'ASM')

        glcm_info = {
            'contrast': float(np.mean(contrast)),
            'dissimilarity': float(np.mean(dissimilarity)),
            'homogeneity': float(np.mean(homogeneity)),
            'energy': float(np.mean(energy)),
            'correlation': float(np.mean(correlation)),
            'asm': float(np.mean(asm))
        }
        
        scaled_color_features = color_scaler.transform([color_features])
        scaled_texture_features = texture_scaler.transform([texture_features])
        
        feature_vector = np.concatenate((scaled_color_features, scaled_texture_features), axis=1)
        
        prediction = svm_model.predict(feature_vector)[0]
        probabilities = svm_model.predict_proba(feature_vector)[0]
        
        class_names = ['Normal', 'Cataract']
        
        result = class_names[prediction]
        confidence = probabilities[prediction] * 100
        
        # Debug info
        print(f"Prediction: {prediction}")
        print(f"Probabilities: Normal={probabilities[0]:.4f}, Cataract={probabilities[1]:.4f}")
        print(f"Result: {result}")
        print(f"Confidence: {confidence:.2f}%")
        
        return {
            'result': result,
            'confidence': confidence,
            'glcm_features': glcm_info,
            'probabilities': {
                'normal': float(probabilities[0]),
                'cataract': float(probabilities[1])
            }
        }, None
        
    except Exception as e:
        return None, f"Error during prediction: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    if file and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                return jsonify({'success': False, 'message': 'Could not decode image'})
            
            prediction_data, error = predict_image_from_array(img)
            
            if prediction_data:
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                img_data_url = f"data:image/jpeg;base64,{img_base64}"
                
                return jsonify({
                    'success': True, 
                    'prediction': prediction_data['result'], 
                    'confidence': f"Confidence: {prediction_data['confidence']:.1f}%",
                    'image_url': img_data_url,
                    'glcm_features': prediction_data['glcm_features'],
                    'probabilities': prediction_data['probabilities']
                })
            else:
                return jsonify({'success': False, 'message': error})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error processing image: {str(e)}'})
    
    return jsonify({'success': False, 'message': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True)
