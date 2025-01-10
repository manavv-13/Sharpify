document.getElementById('file-input').addEventListener('change', function () {
    const fileName = this.files[0] ? this.files[0].name : 'No file selected';
    const fileNameElement = document.getElementById('file-name');
    const uploadText = document.getElementById('upload-text');
    
    fileNameElement.textContent = fileName; // Set the selected file name
    fileNameElement.style.display = 'block'; // Show the file name
    uploadText.style.display = 'none'; // Hide the original text
});

document.getElementById('image-form').addEventListener('submit', function (e) {
    // Show the loader overlay
    const loader = document.getElementById('loader-overlay');
    loader.style.display = 'flex';
});
