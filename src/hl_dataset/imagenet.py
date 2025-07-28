import torch.utils.data as data
from datasets import load_dataset

class ImageNetDataset(data.Dataset):
    def __init__(self, split, transform=None, max_len=None):
        self.split = split
        self.transform = transform
        self.max_len = max_len

        # Only one dataset should be loaded — keeping the last one
        self.dataset = load_dataset("animetimm/danbooru-wdtagger-v4-w640-ws-150k", split=split)

    def __getitem__(self, index):
        entry = self.dataset[index]

        # Use appropriate keys for the selected dataset
        img = entry["webp"]
        target = entry["__key__"]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.max_len or len(self.dataset)
