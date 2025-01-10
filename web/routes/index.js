const express = require('express');
const router = express.Router();
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');


// Home Page
router.get('/', (req, res) => {
    res.render('index');
});

// Image Upload Page
router.get('/uploadDeblur', (req, res) => {
    res.render('upload', { inputImage: null, outputImage: null });
});
router.get('/uploadTranslate', (req, res) => {
    res.render('translate');
});
// Process Image
// Process Image
router.post('/process', (req, res) => {
    if (!req.files || !req.files.image) {
        return res.status(400).send('No image file uploaded.');
    }

    const image = req.files.image;
    const uploadDir = path.join(__dirname, '../uploads');
    const inputPath = path.join(uploadDir, image.name);
    const outputPath = path.join(uploadDir, `deblurred_${image.name}`);

    // Save the uploaded file
    image.mv(inputPath, (err) => {
        if (err) return res.status(500).send('Error saving the uploaded file.');

        // Spawn Python process
        const python = spawn('python', [
            path.join(__dirname, '../../training/test_deblur.py'),
            '--input', inputPath,
            '--output', outputPath,
        ]);

        python.stdout.on('data', (data) => {
            console.log(`Python stdout: ${data}`);
        });

        python.stderr.on('data', (data) => {
            console.error(`Python stderr: ${data}`);
        });

        python.on('close', (code) => {
            console.log(`Python process exited with code ${code}`);
            if (code === 0) {
                // Successfully deblurred
                res.render('upload', {
                    inputImage: `/uploads/${image.name}`,
                    outputImage: `/uploads/deblurred_${image.name}`,
                });
            } else {
                res.status(500).send('Error during deblurring process.');
            }
        });
    });
});

module.exports = router;
