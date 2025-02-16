import torch.utils.data as data
from datasets import load_dataset


class ImageNetDataset(data.Dataset):
    def __init__(self, split, transform=None):
        self.split = split
        self.transform = transform
        self.dataset = load_dataset("evanarlian/imagenet_1k_resized_256", split=split)

    def __getitem__(self, index):
        entry = self.dataset[index]
        img = entry["image"]
        target = entry["label"]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.dataset)
