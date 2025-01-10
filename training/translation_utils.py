import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.files_X = sorted(os.listdir(os.path.join(root_dir, "domain_x")))
        self.files_Y = sorted(os.listdir(os.path.join(root_dir, "domain_y")))
        self.root_X = os.path.join(root_dir, "domain_x")
        self.root_Y = os.path.join(root_dir, "domain_y")

    def __len__(self):
        return max(len(self.files_X), len(self.files_Y))

    def __getitem__(self, idx):
        file_X = os.path.join(self.root_X, self.files_X[idx % len(self.files_X)])
        file_Y = os.path.join(self.root_Y, self.files_Y[idx % len(self.files_Y)])
        img_X = Image.open(file_X).convert("RGB")
        img_Y = Image.open(file_Y).convert("RGB")
        if self.transform:
            img_X = self.transform(img_X)
            img_Y = self.transform(img_Y)
        return img_X, img_Y

def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
