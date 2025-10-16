import torch.utils.data as data
from datasets import load_dataset


class ImageNetDataset(data.Dataset):
    def __init__(self, split, transform=None, max_len=None):
        self.split = split
        self.transform = transform
        self.dataset = load_dataset("animetimm/danbooru-wdtagger-v4-w640-ws-50k", split=split)
        self.max_len = max_len

    def __getitem__(self, index):
        entry = self.dataset[index]
        img = entry["webp"]
        target = entry["json"]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.max_len or len(self.dataset)