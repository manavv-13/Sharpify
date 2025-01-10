import os
import cv2
import torch
import argparse
from torchvision.transforms import transforms
from model import DeblurCNN
import numpy as np

# Argument parsing for input and output paths
parser = argparse.ArgumentParser(description="Image Deblurring Script")
parser.add_argument('--input', type=str, required=True, help="Path to the input blurred image")
parser.add_argument('--output', type=str, required=True, help="Path to save the deblurred image")
parser.add_argument('--iterations', type=int, default=3, help="Number of deblurring iterations (default: 3)")
args = parser.parse_args()

# Paths and configurations
MODEL_PATH = "../models/best_deblur_cnn.pth"
INPUT_IMAGE_PATH = args.input
OUTPUT_IMAGE_PATH = args.output
NUM_ITERATIONS = args.iterations

# Device configuration
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the model
try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model = DeblurCNN().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Image preprocessing
def preprocess_image(image, image_size=512):
    try:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),  # Resize to 512x512 for the model
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])  # No normalization
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        exit(1)

# Postprocessing and saving the output
def save_deblurred_image(tensor, output_path, original_size):
    try:
        # Convert tensor to numpy array
        tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Clip values to [0, 1] and scale to 0-255
        tensor = np.clip(tensor, 0, 1)
        tensor = (tensor * 255).astype(np.uint8)

        # Convert to BGR for saving and resize to original dimensions
        tensor = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
        tensor = cv2.resize(tensor, (original_size[1], original_size[0]))  # Resize to original (width, height)

        # Save the image
        cv2.imwrite(output_path, tensor)
        print(f"Deblurred image saved at: {output_path}")
    except Exception as e:
        print(f"Error in saving deblurred image: {e}")
        exit(1)

# Load and preprocess the input image
try:
    input_image = cv2.imread(INPUT_IMAGE_PATH)
    if input_image is None:
        raise ValueError("Image not found or cannot be read.")
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    original_size = input_image.shape  # Save the original size for later use
    print(f"Original input image size: {original_size}")

    input_tensor = preprocess_image(input_image).to(device)
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor min: {input_tensor.min()}, max: {input_tensor.max()}")
except Exception as e:
    print(f"Error loading input image: {e}")
    exit(1)

# Deblur the image iteratively
try:
    current_tensor = input_tensor
    for i in range(NUM_ITERATIONS):
        with torch.no_grad():
            current_tensor = model(current_tensor)
            current_tensor = torch.clamp(current_tensor, 0, 1)  # Ensure values are within [0, 1]
        print(f"Iteration {i + 1}: Tensor min: {current_tensor.min()}, max: {current_tensor.max()}")

    # Save the final deblurred image
    save_deblurred_image(current_tensor, OUTPUT_IMAGE_PATH, original_size)
except Exception as e:
    print(f"Error during deblurring process: {e}")
    exit(1)
