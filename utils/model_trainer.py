import numpy as np
import pickle
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from .data_loader import load_dataset_from_folders

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

def train_model():
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
    
    for test_size, split_ratio in zip(test_sizes, split_ratios):
        print(f"\nMenguji pembagian train-test {split_ratio}...")
        
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

        for kernel_name, param_grid in param_grids.items():
            print(f"  Menguji kernel {kernel_name.upper()}...")

            svm = SVC(probability=True, random_state=42)
            
            grid_search = GridSearchCV(
                svm, param_grid, 
                cv=skf,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            try:
                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                
                y_pred = best_model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=skf, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                train_accuracy = best_model.score(X_train, y_train)
                overfitting_gap = train_accuracy - test_accuracy
                
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
    
    if best_overall_model is None:
        return False, "No valid model could be trained. Please check your dataset."
    
    # Simpan model dan semua komponen preprocessing
    model_data = {
        'model': best_overall_model,
        'color_scaler': color_scaler,
        'texture_scaler': texture_scaler,
        'pca_reducer': pca_reducer,
        'best_params': best_overall_params,
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
    
    results_html = generate_training_results_html(results_summary, best_overall_params, time.time() - start_time, model_data['feature_dimensions'])
    
    return True, results_html
