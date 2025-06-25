from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64

from utils.model_trainer import train_model
from utils.predictor import predict_image_core
from utils.zip_handler import extract_dataset_zip, validate_zip_file, get_dataset_info

app = Flask(__name__)

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

@app.route('/training')
def train_page():
    return render_template('train.html')

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

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    try:
        if 'dataset_zip' not in request.files:
            return jsonify({'success': False, 'message': 'Tidak ada file ZIP yang diupload'})
        
        zip_file = request.files['dataset_zip']
        
        # Validate ZIP file
        is_valid, result = validate_zip_file(zip_file)
        if not is_valid:
            return jsonify({'success': False, 'message': result})
        
        # Extract dataset
        success, message = extract_dataset_zip(zip_file)
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/dataset_info', methods=['GET'])
def dataset_info():
    try:
        info = get_dataset_info()
        if info is None:
            return jsonify({'exists': False})
        return jsonify(info)
    except Exception as e:
        return jsonify({'exists': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
