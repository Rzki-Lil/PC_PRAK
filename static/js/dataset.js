function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    document.getElementById(tabName + '-tab').classList.add('active');
    
    event.target.classList.add('active');
}

window.addEventListener('load', function() {
    checkDatasetInfo();
});

document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('zipFile').addEventListener('change', function(e) {
        const file = e.target.files[0];
        const fileName = document.getElementById('zipFileName');
        const uploadBtn = document.getElementById('uploadZipBtn');
        
        if (file) {
            fileName.textContent = `📁 ${file.name} (${formatFileSize(file.size)})`;
            fileName.style.display = 'block';
            uploadBtn.style.display = 'inline-block';
        } else {
            fileName.style.display = 'none';
            uploadBtn.style.display = 'none';
        }
    });
});

function uploadDataset() {
    const fileInput = document.getElementById('zipFile');
    const uploadBtn = document.getElementById('uploadZipBtn');
    
    if (!fileInput.files[0]) {
        Swal.fire({
            toast: true,
            position: 'top',
            icon: 'warning',
            title: '📦 Silakan pilih file ZIP terlebih dahulu!',
            showConfirmButton: false,
            timer: 3000,
            timerProgressBar: true,
            width: '400px'
        });
        return;
    }
    
    const formData = new FormData();
    formData.append('dataset_zip', fileInput.files[0]);
    
    uploadBtn.disabled = true;
    uploadBtn.textContent = '📤 Mengupload...';
    
    fetch('/upload_dataset', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            Swal.fire({
                toast: true,
                position: 'top',
                icon: 'success',
                title: data.message,
                showConfirmButton: false,
                timer: 4000,
                timerProgressBar: true,
                width: '500px'
            });
            checkDatasetInfo();
            document.getElementById('zipFile').value = '';
            document.getElementById('zipFileName').style.display = 'none';
            uploadBtn.style.display = 'none';
        } else {
            Swal.fire({
                toast: true,
                position: 'top',
                icon: 'error',
                title: data.message,
                showConfirmButton: false,
                timer: 5000,
                timerProgressBar: true,
                width: '500px'
            });
        }
    })
    .catch(error => {
        Swal.fire({
            toast: true,
            position: 'top',
            icon: 'error',
            title: `Error: ${error.message}`,
            showConfirmButton: false,
            timer: 5000,
            timerProgressBar: true,
            width: '500px'
        });
    })
    .finally(() => {
        uploadBtn.disabled = false;
        uploadBtn.textContent = '📤 Upload Dataset';
    });
}

function checkDatasetInfo() {
    fetch('/dataset_info')
    .then(response => response.json())
    .then(data => {
        const datasetInfo = document.getElementById('datasetInfo');
        if (data.exists) {
            datasetInfo.innerHTML = `
                <div class="status-indicator success">
                    ✅ Dataset siap! Normal: ${data.normal_count} gambar, Cataract: ${data.cataract_count} gambar
                </div>
            `;
        } else {
            datasetInfo.innerHTML = `
                <div class="status-indicator warning">
                    ⚠️ Dataset belum tersedia. Upload file ZIP dataset terlebih dahulu.
                </div>
            `;
        }
    })
    .catch(error => {
        console.error('Error checking dataset info:', error);
    });
}



function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
