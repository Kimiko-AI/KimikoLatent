import torch.utils.data as data
from datasets import load_dataset
from torchvision import transforms
import torchvision.transforms.functional as F


class ImageNetDataset(data.Dataset):
    def __init__(self, split, tran, max_len=None):
        self.split = split
        self.max_len = max_len

        self.dataset = load_dataset(
            "animetimm/danbooru-wdtagger-v4-w640-ws-1M",
            split=split
        )

        # Shared crop transform (resize shortest edge + random crop + resize to 256)
        self.shared_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(512, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC)
        ])

        # Post-processing for train
        self.train_post = transforms.Compose([

            transforms.ToTensor(),
        ])

        # Post-processing for DINO
        self.dino_post = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC)
        ])

    def __getitem__(self, index):
        entry = self.dataset[index]
        img = entry["webp"]

        # Apply shared crop once
        crop = self.shared_transform(img)

        # Apply separate post-processing
        train_img = self.train_post(crop) * 2 - 1
        dino_img = self.dino_post(crop) * 2 - 1

        return dino_img, train_img

    def __len__(self):
        return self.max_len or len(self.dataset)
