import torch.utils.data as data
from datasets import load_dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import random

def process(x):
    return x * 2 - 1

from torchvision import transforms

class ImageNetDataset(data.Dataset):
    def __init__(self, split, transform=None, max_len=None):
        self.split = split
        self.max_len = max_len

        self.dataset = load_dataset(
            "animetimm/danbooru-wdtagger-v4-w640-ws-50k",
            split=split
        )

        # Post-processing for train (256x256)
        self.train_post = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

        # Post-processing for DINO
        self.dino_post = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        entry = self.dataset[index]
        img = entry["webp"]

        # Step 1: resize shortest edge to 518, keep aspect ratio
        w, h = img.size
        if w < h:
            new_w = 512
            new_h = int(h * (512 / w))
        else:
            new_h = 512
            new_w = int(w * (512 / h))
        img = F.resize(img, (new_h, new_w), interpolation=transforms.InterpolationMode.BICUBIC)

        # Step 2: random crop from resized image
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(512, 512))
        dino_crop = F.crop(img, i, j, h, w)
        train_crop = dino_crop.copy()  # same crop for train

        # Step 3: resize train crop to 256×256
        train_crop = F.resize(train_crop, (256, 256), interpolation=transforms.InterpolationMode.BICUBIC)

        # Step 4: apply transforms
        img_proc = self.train_post(train_crop)
        dino_proc = self.dino_post(dino_crop)
        img_proc = img_proc * 2 - 1
        return img_proc, dino_proc

    def __len__(self):
        return self.max_len or len(self.dataset)