function trainModel() {
    const trainBtn = document.getElementById('trainBtn');
    const trainResult = document.getElementById('trainResult');
    
    trainBtn.disabled = true;
    trainBtn.textContent = '🔄 Melatih...';
    trainResult.innerHTML = '';
    
    fetch('/train', {
        method: 'POST'
    })
    .then(response => response.json())
    .then((data) => {
        trainResult.innerHTML = `<div class="result ${data.success ? 'success' : 'error'}">
            ${data.message}
        </div>`;
    })
    .catch(error => {
        trainResult.innerHTML = `<div class="result error">❌ Kesalahan: ${error}</div>`;
    })
    .finally(() => {
        trainBtn.disabled = false;
        trainBtn.textContent = '🚀 Mulai Training';
    });
}

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('imageFile');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const imageInfo = document.getElementById('imageInfo');

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    if (!file.type.match('image.*')) {
        Swal.fire({
            toast: true,
            position: 'top',
            icon: 'error',
            title: '❌ Silakan pilih file gambar yang valid!',
            showConfirmButton: false,
            timer: 3000,
            timerProgressBar: true,
            width: '400px'
        });
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        imageInfo.textContent = `📁 ${file.name} (${formatFileSize(file.size)})`;
        
        document.querySelector('.upload-content').style.display = 'none';
        imagePreview.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

async function pasteFromClipboard() {
    try {
        const clipboardItems = await navigator.clipboard.read();
        for (const clipboardItem of clipboardItems) {
            for (const type of clipboardItem.types) {
                if (type.startsWith('image/')) {
                    const blob = await clipboardItem.getType(type);
                    const file = new File([blob], 'gambar-tempel.png', { type: blob.type });
                    
                    const dt = new DataTransfer();
                    dt.items.add(file);
                    fileInput.files = dt.files;
                    
                    handleFileSelect(file);
                    return;
                }
            }
        }
        Swal.fire({
            toast: true,
            position: 'top',
            icon: 'info',
            title: '📋 Tidak ada gambar di clipboard!',
            showConfirmButton: false,
            timer: 3000,
            timerProgressBar: true,
            width: '400px'
        });
    } catch (err) {
        Swal.fire({
            toast: true,
            position: 'top',
            icon: 'error',
            title: '❌ Gagal mengakses clipboard. Silakan gunakan seret & lepas atau pilih file.',
            showConfirmButton: false,
            timer: 4000,
            timerProgressBar: true,
            width: '450px'
        });
    }
}

function removeImage() {
    fileInput.value = '';
    imagePreview.style.display = 'none';
    document.querySelector('.upload-content').style.display = 'flex';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function predictImage() {
    const predictionResult = document.getElementById('predictionResult');
    const loading = document.getElementById('loading');
    
    if (!fileInput.files[0]) {
        Swal.fire({
            toast: true,
            position: 'top',
            icon: 'warning',
            title: '📷 Silakan pilih file gambar terlebih dahulu!',
            showConfirmButton: false,
            timer: 3000,
            timerProgressBar: true,
            width: '400px'
        });
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    loading.style.display = 'block';
    predictionResult.innerHTML = '';
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const emoji = data.prediction === 'Normal' ? '✅' : '⚠️';
            
            const glcmFeaturesHTML = Object.entries(data.glcm_features).map(([key, value]) => `
                <div class="feature-item-inline">
                    <div class="feature-name-value">
                        <span class="feature-name">${key.toUpperCase()}</span>
                        <span class="feature-value">${value.toFixed(4)}</span>
                    </div>
                </div>
            `).join('');
            
            predictionResult.innerHTML = `
                <div class="prediction-result-new">
                    <div class="top-section">
                        <div class="prediction-image-container">
                            <img src="${data.image_url}" alt="Gambar yang Diunggah" class="prediction-image">
                        </div>
                        
                        <div class="prediction-info">
                            <div class="prediction-result-card">
                                <h3>${emoji} Hasil Prediksi</h3>
                                <div style="font-size: 1.1rem; margin-bottom: 10px;">
                                    <strong>Diagnosis: ${data.prediction}</strong>
                                </div>
                                <div style="margin-bottom: 15px;">
                                    🎯 Tingkat Kepercayaan: ${data.confidence.replace('Confidence:', '')}
                                </div>
                                
                                <div class="probability-bars">
                                    <div class="probability-item">
                                        <div class="probability-label">
                                            <span>Normal</span>
                                            <span>${(data.probabilities.normal * 100).toFixed(1)}%</span>
                                        </div>
                                        <div class="probability-bar">
                                            <div class="probability-fill probability-normal" 
                                                 style="width: ${data.probabilities.normal * 100}%"></div>
                                        </div>
                                    </div>
                                    
                                    <div class="probability-item">
                                        <div class="probability-label">
                                            <span>Katarak</span>
                                            <span>${(data.probabilities.cataract * 100).toFixed(1)}%</span>
                                        </div>
                                        <div class="probability-bar">
                                            <div class="probability-fill probability-cataract" 
                                                 style="width: ${data.probabilities.cataract * 100}%"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="glcm-features-section">
                        <div class="glcm-features-card-full">
                            <h4>📈 Fitur Tekstur GLCM</h4>
                            <div class="glcm-features-grid">
                                ${glcmFeaturesHTML}
                            </div>
                        </div>
                        
                        <div class="glcm-explanation-card">
                            <h4>📚 Penjelasan Fitur GLCM</h4>
                            <div class="explanation-content">
                                <div class="explanation-item">
                                    <h5>🔹 ASM (Angular Second Moment)</h5>
                                    <p>Mengukur keseragaman tekstur dalam gambar. Nilai tinggi menunjukkan distribusi intensitas yang seragam dan tekstur yang halus. Pada mata katarak, nilai ASM cenderung berbeda karena adanya kekeruhan yang mengubah pola tekstur normal mata.</p>
                                </div>
                                
                                <div class="explanation-item">
                                    <h5>🔹 Contrast (Kontras)</h5>
                                    <p>Mengukur perbedaan intensitas antara piksel tetangga. Nilai tinggi menunjukkan variasi intensitas yang besar dalam gambar. Katarak dapat mengubah kontras mata karena menghalangi cahaya dan menciptakan pola bayangan yang tidak normal.</p>
                                </div>
                                
                                <div class="explanation-item">
                                    <h5>🔹 Correlation (Korelasi)</h5>
                                    <p>Mengukur ketergantungan linear antara piksel tetangga. Nilai tinggi menunjukkan hubungan yang kuat antar piksel. Mata normal memiliki pola korelasi yang teratur, sedangkan katarak dapat mengacaukan pola tersebut.</p>
                                </div>
                                
                                <div class="explanation-item">
                                    <h5>🔹 Dissimilarity (Ketidaksamaan)</h5>
                                    <p>Mengukur ketidaksamaan antara piksel tetangga. Nilai tinggi menunjukkan tekstur yang kasar dan tidak seragam. Katarak sering menyebabkan peningkatan dissimilarity karena menciptakan pola yang tidak teratur pada mata.</p>
                                </div>
                                
                                <div class="explanation-item">
                                    <h5>🔹 Energy (Energi)</h5>
                                    <p>Mengukur keseragaman distribusi intensitas dalam gambar. Nilai tinggi menunjukkan distribusi yang seragam. Mata sehat umumnya memiliki energi yang lebih stabil dibandingkan mata dengan katarak yang memiliki distribusi intensitas tidak merata.</p>
                                </div>
                                
                                <div class="explanation-item">
                                    <h5>🔹 Homogeneity (Homogenitas)</h5>
                                    <p>Mengukur kedekatan distribusi elemen GLCM terhadap diagonal GLCM. Nilai tinggi menunjukkan tekstur yang halus dan seragam. Katarak menurunkan homogenitas karena menciptakan area dengan intensitas yang bervariasi secara tidak teratur.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            // Show error message using SweetAlert toast
            Swal.fire({
                toast: true,
                position: 'top',
                icon: 'error',
                title: data.message,
                showConfirmButton: false,
                timer: 4000,
                timerProgressBar: true,
                width: '500px'
            });
            predictionResult.innerHTML = '';
        }
    })
    .catch(error => {
        // Show error message using SweetAlert toast
        Swal.fire({
            toast: true,
            position: 'top',
            icon: 'error',
            title: `Terjadi kesalahan: ${error}`,
            showConfirmButton: false,
            timer: 4000,
            timerProgressBar: true,
            width: '450px'
        });
        predictionResult.innerHTML = '';
    })
    .finally(() => {
        loading.style.display = 'none';
    });
}
