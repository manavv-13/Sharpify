import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split

def prepare_data(data_dir, test_size=0.25, image_size=512, batch_size=4, use_gopro=False):
    if use_gopro:
        # GoPro dataset paths
        gopro_blur_dir = os.path.join(data_dir, "train/blur")
        gopro_sharp_dir = os.path.join(data_dir, "train/sharp")

        # Load file lists for GoPro dataset
        gopro_blur_files = sorted([f for f in os.listdir(gopro_blur_dir) if f.endswith(".png") or f.endswith(".jpg")])
        gopro_sharp_files = sorted([f for f in os.listdir(gopro_sharp_dir) if f.endswith(".png") or f.endswith(".jpg")])

        # Ensure matching filenames
        gopro_pairs = [(os.path.join(gopro_blur_dir, f), os.path.join(gopro_sharp_dir, f)) for f in gopro_blur_files if f in gopro_sharp_files]

        blur_paths, sharp_paths = zip(*gopro_pairs)
    else:
        # Existing datasets
        motion_blur_dir = os.path.join(data_dir, "motion_blurred")
        defocus_blur_dir = os.path.join(data_dir, "defocused_blurred")
        sharp_dir = os.path.join(data_dir, "sharp")

        # Load file lists for existing datasets
        motion_files = sorted([f for f in os.listdir(motion_blur_dir) if f.endswith("_M.jpg")])
        defocus_files = sorted([f for f in os.listdir(defocus_blur_dir) if f.endswith("_F.jpg")])
        sharp_files = sorted([f for f in os.listdir(sharp_dir) if f.endswith("_S.jpg")])

        motion_pairs = [(os.path.join(motion_blur_dir, f), os.path.join(sharp_dir, f.replace("_M", "_S"))) for f in motion_files]
        defocus_pairs = [(os.path.join(defocus_blur_dir, f), os.path.join(sharp_dir, f.replace("_F", "_S"))) for f in defocus_files]

        all_pairs = motion_pairs + defocus_pairs
        blur_paths, sharp_paths = zip(*all_pairs)

    # Train-test split
    train_blur, val_blur, train_sharp, val_sharp = train_test_split(blur_paths, sharp_paths, test_size=test_size)
    
    print(f"Train samples: {len(train_blur)}, Validation samples: {len(val_blur)}")

    # Data Transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    class DeblurDataset(Dataset):
        def __init__(self, blur_paths, sharp_paths, transforms):
            self.blur_paths = blur_paths
            self.sharp_paths = sharp_paths
            self.transforms = transforms
        
        def __len__(self):
            return len(self.blur_paths)
        
        def __getitem__(self, idx):
            blur_image = cv2.imread(self.blur_paths[idx])
            blur_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB)
            sharp_image = cv2.imread(self.sharp_paths[idx])
            sharp_image = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2RGB)
            if self.transforms:
                blur_image = self.transforms(blur_image)
                sharp_image = self.transforms(sharp_image)
            return blur_image, sharp_image

    train_data = DeblurDataset(train_blur, train_sharp, transform)
    val_data = DeblurDataset(val_blur, val_sharp, transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
