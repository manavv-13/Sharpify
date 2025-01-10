import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from model import DeblurCNN
from data_loader import prepare_data

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=40, help='Number of epochs to train')
parser.add_argument('--data_dir', type=str, default="../data/deblurring", help='Path to the data directory')
parser.add_argument('--image_size', type=int, default=512, help='Size to resize images during training')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
parser.add_argument('--load_model', type=str, default=None, help='Path to pre-trained model to continue training')
parser.add_argument('--use_gopro', action='store_true', help='Train only on the GoPro dataset')
args = vars(parser.parse_args())

# Device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# Data Loaders
train_loader, val_loader = prepare_data(
    data_dir=args['data_dir'], 
    image_size=args['image_size'], 
    batch_size=args['batch_size'],
    use_gopro=args['use_gopro']
)

# Model
model = DeblurCNN().to(device)

# If a pre-trained model is provided, load its weights
if args['load_model']:
    print(f"Loading pre-trained model from {args['load_model']}")
    model.load_state_dict(torch.load(args['load_model'], map_location=device))

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and Validation Functions
def train_one_epoch(model, loader, optimizer):
    model.train()
    running_loss = 0.0
    for blur, sharp in tqdm(loader):
        blur, sharp = blur.to(device), sharp.to(device)
        optimizer.zero_grad()
        outputs = model(blur)
        loss = criterion(outputs, sharp)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate_one_epoch(model, loader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for blur, sharp in tqdm(loader):
            blur, sharp = blur.to(device), sharp.to(device)
            outputs = model(blur)
            loss = criterion(outputs, sharp)
            running_loss += loss.item()
    return running_loss / len(loader)

# Main Training Loop
epochs = args['epochs']
train_losses, val_losses = [], []
best_val_loss = float('inf')  # Initialize the best validation loss

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = validate_one_epoch(model, val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "../models/best_deblur_cnn.pth")
        print("Best model saved!")

# Summary of Training
print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
