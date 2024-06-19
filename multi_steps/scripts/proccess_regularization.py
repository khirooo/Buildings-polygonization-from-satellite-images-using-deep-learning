import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
from models.networks import GenModel
from utils.helpers import load_model

class RegularizationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def regularize_images(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            outputs = model(images)
            save_images(outputs, dataloader.dataset.image_files, 'regularized_images', device)

def save_images(images, image_names, output_dir, device):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    images = images.cpu().numpy()
    for i in range(images.shape[0]):
        img = transforms.ToPILImage()(images[i])
        img.save(os.path.join(output_dir, image_names[i]))

def main():
    # Define parameters
    image_dir = 'path/to/images'
    checkpoint_path = 'path/to/checkpoint.pth'
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load dataset
    dataset = RegularizationDataset(image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = GenModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    epoch = load_model(checkpoint_path, model, optimizer)

    # Regularize images
    regularize_images(model, dataloader, device)

if __name__ == "__main__":
    main()

