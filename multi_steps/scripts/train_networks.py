import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.networks import GenModel, DiscModel
from data.loader import create_data_loader
from utils.helpers import save_model, load_model
import os

def train():
    # Hardcoded parameters
    image_dir = 'data/train_images'
    batch_size = 32
    num_epochs = 100
    checkpoint_dir = 'checkpoints'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data loader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataloader = create_data_loader(image_dir, batch_size, transform=transform)

    # Initialize models
    generator = GenModel().to(device)
    discriminator = DiscModel().to(device)

    # Initialize optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(num_epochs):
        for i, images in enumerate(dataloader):
            # Prepare data
            real_images = images.to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            d_optimizer.zero_grad()
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)
            d_loss_real.backward()

            fake_images = generator(torch.randn(batch_size, 3, 256, 256).to(device))
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss_fake.backward()
            d_optimizer.step()

            d_loss = d_loss_real + d_loss_fake

            # Train Generator
            g_optimizer.zero_grad()
            fake_images = generator(torch.randn(batch_size, 3, 256, 256).to(device))
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()

            # Print losses
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], "
                      f"D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

        # Save checkpoints
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        save_model(generator, g_optimizer, epoch, os.path.join(checkpoint_dir, f'generator_epoch_{epoch}.pth'))
        save_model(discriminator, d_optimizer, epoch, os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch}.pth'))

if __name__ == "__main__":
    train()

