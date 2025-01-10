import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from cycle_gan import Generator, Discriminator
from translation_utils import ImageDataset, save_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_cycle_gan(data_dir, epochs=200, batch_size=1, lr=0.0002, lambda_cycle=10.0):
    # Initialize generators and discriminators
    G_XtoY = Generator(3, 3).to(device)
    G_YtoX = Generator(3, 3).to(device)
    D_X = Discriminator(3).to(device)
    D_Y = Discriminator(3).to(device)

    # Optimizers
    optim_G = torch.optim.Adam(list(G_XtoY.parameters()) + list(G_YtoX.parameters()), lr=lr, betas=(0.5, 0.999))
    optim_D_X = torch.optim.Adam(D_X.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_D_Y = torch.optim.Adam(D_Y.parameters(), lr=lr, betas=(0.5, 0.999))

    # Losses
    adversarial_loss = torch.nn.MSELoss()
    cycle_loss = torch.nn.L1Loss()

    # Datasets and dataloaders
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        for i, (real_X, real_Y) in enumerate(dataloader):
            real_X, real_Y = real_X.to(device), real_Y.to(device)

            # Generator forward pass
            fake_Y = G_XtoY(real_X)
            fake_X = G_YtoX(real_Y)

            rec_X = G_YtoX(fake_Y)
            rec_Y = G_XtoY(fake_X)

            # Generator loss
            loss_G = (
                adversarial_loss(D_Y(fake_Y), torch.ones_like(D_Y(fake_Y))) +
                adversarial_loss(D_X(fake_X), torch.ones_like(D_X(fake_X))) +
                lambda_cycle * (cycle_loss(rec_X, real_X) + cycle_loss(rec_Y, real_Y))
            )
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            # Discriminator X loss
            loss_D_X = 0.5 * (
                adversarial_loss(D_X(real_X), torch.ones_like(D_X(real_X))) +
                adversarial_loss(D_X(fake_X.detach()), torch.zeros_like(D_X(fake_X)))
            )
            optim_D_X.zero_grad()
            loss_D_X.backward()
            optim_D_X.step()

            # Discriminator Y loss
            loss_D_Y = 0.5 * (
                adversarial_loss(D_Y(real_Y), torch.ones_like(D_Y(real_Y))) +
                adversarial_loss(D_Y(fake_Y.detach()), torch.zeros_like(D_Y(fake_Y)))
            )
            optim_D_Y.zero_grad()
            loss_D_Y.backward()
            optim_D_Y.step()

            print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}]: "
                  f"Loss_G: {loss_G.item()}, Loss_D_X: {loss_D_X.item()}, Loss_D_Y: {loss_D_Y.item()}")

        # Save checkpoints
        save_checkpoint(G_XtoY, "G_XtoY.pth")
        save_checkpoint(G_YtoX, "G_YtoX.pth")

if __name__ == "__main__":
    train_cycle_gan(data_dir="data/translation", epochs=100)
