from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class MMOTUSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, split_file, image_size=(256, 256)):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        with open(split_file, "r") as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]

        image_path = self.image_dir / f"{image_id}.JPG"
        mask_path = self.mask_dir / f"{image_id}.PNG"

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.image_size)
        mask = mask.resize(self.image_size, Image.Resampling.NEAREST)

        image = np.asarray(image, dtype=np.float32) / 255.0
        mask = np.asarray(mask, dtype=np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        mask = (mask > 0).float()

        return image, mask