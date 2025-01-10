import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from cycle_gan import Generator  # Import your Generator class

def load_model(checkpoint_path, device):
    """Load a pretrained generator model."""
    model = Generator().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

def translate_image(input_image_path, output_image_path, model_path, device):
    """Translate a single image using the trained CycleGAN model."""
    # Load the pretrained generator model
    model = load_model(model_path, device)

    # Transformation for input images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to the size used during training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
    ])

    # Load and preprocess the input image
    image = Image.open(input_image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Generate the translated image
    with torch.no_grad():
        translated_tensor = model(input_tensor)

    # Denormalize the output image
    translated_tensor = translated_tensor.squeeze(0) * 0.5 + 0.5  # Convert to [0, 1]

    # Convert tensor to PIL image
    translated_image = transforms.ToPILImage()(translated_tensor.cpu())

    # Save the translated image
    translated_image.save(output_image_path)
    print(f"Input image: {input_image_path}")
    print(f"Translated image saved to: {output_image_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test a CycleGAN model on a single image.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the translated image.")
    parser.add_argument("--model", type=str, required=True, choices=["XtoY", "YtoX"],
                        help="Specify the model to use: 'XtoY' or 'YtoX'.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference ('cuda' or 'cpu').")

    args = parser.parse_args()

    # Select the model checkpoint based on the direction
    model_path = "models/G_XtoY.pth" if args.model == "XtoY" else "models/G_YtoX.pth"

    # Translate the image
    translate_image(args.input, args.output, model_path, args.device)
