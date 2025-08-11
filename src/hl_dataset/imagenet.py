import torch.utils.data as data
from datasets import load_dataset
import timm
import torch
from torchvision import transforms
class ImageNetDataset(data.Dataset):
    def __init__(self, split, transform=None, max_len=None):
        self.split = split
        self.transform = transform
        self.max_len = max_len

        # Only one dataset should be loaded — keeping the last one
        self.dataset = load_dataset("animetimm/danbooru-wdtagger-v4-w640-ws-150k", split=split)
        self.transforms_dino = transform = transforms.Compose([
    transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),  # redundant here since we resize exactly, but keeps cfg consistency
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

    def __getitem__(self, index):
        entry = self.dataset[index]

        # Use appropriate keys for the selected dataset
        img = entry["webp"]
        target = entry["__key__"]
        dino = self.transforms_dino(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, dino

    def __len__(self):
        return self.max_len or len(self.dataset)
