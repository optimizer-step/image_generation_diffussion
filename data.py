"""CHAPTER 1: Give the model real pet photos."""
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from config import IMAGE_SIZE


def make_loader(batch_size):
    photo_prep = transforms.Compose([
        transforms.Lambda(lambda photo: photo.convert("RGB")),
        transforms.Resize(80),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    pets = OxfordIIITPet(
        root="data", split="trainval", download=True, transform=photo_prep
    )
    loader = DataLoader(pets, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)
    return loader
