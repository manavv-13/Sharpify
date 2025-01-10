# Sharpify
Sharpify is an AI-driven software solution for image enhancement, focusing on image deblurring and transformation. By leveraging cutting-edge technologies like Convolutional Neural Networks (CNNs) for deblurring and CycleGAN for image translation, Sharpify restores clarity to blurry images and transforms visuals seamlessly.

## Features
Image Deblurring: Removes motion blur, defocus blur, and other distortions to enhance image sharpness.
Image Translation: Converts images between different domains or styles (e.g., grayscale to color). (**Feature not works yet, due to incompatible device**)
User-Friendly Interface: Intuitive platform for users of all technical levels.
Scalable and Secure: Handles images of varying complexities while ensuring data privacy.

## Tech Stack
Programming Language:
### Python: Backbone of the project, used for model implementation and backend logic.

### Frameworks and Libraries:

PyTorch: Deep learning framework for training and deploying CNNs and CycleGANs.

torch.nn for defining neural networks.
torch.optim for optimization algorithms.
torch.utils.data for dataset handling.

OpenCV: For image preprocessing, such as resizing and noise reduction.

Torchvision: Data augmentation and image transformation.

Scikit-learn: Dataset management and splitting.

Tqdm: Visual progress tracking during model training and validation.

Argparse: Command-line argument parsing for configurable training parameters.

Pillow (PIL): Image format conversions and preprocessing.


### Web Technologies:
Frontend: HTML, CSS, Bootstrap for a responsive and intuitive UI.
Backend: Node.js and Express.js for handling API requests and integration with ML models.

### Other Tools:
Database: MongoDB for storing metadata and processed images.

## Use Cases
Photography: Enhances image clarity for personal or professional use.
Medical Imaging: Improves diagnostic accuracy by sharpening blurred scans.
Security: Enhances surveillance footage for better identification.
E-commerce: Elevates product presentation through image quality improvement.

## How to Use
Upload Image: Drag and drop or select an image for processing.
Select Process: Choose between image deblurring or translation.
Download: Save the processed image for your needs.


**There are two more folders that I haven't uploaded - **data** and **testing** which contained dataset which I used and testing images which I used to test the accuracy of model**
