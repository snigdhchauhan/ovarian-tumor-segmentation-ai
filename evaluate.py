import torch
from torch.utils.data import DataLoader

from preprocessing.dataset import MMOTUSegmentationDataset
from models.unet import UNet


IMAGE_DIR = r"data\MMOTU\MMOTU\images"
MASK_DIR = r"data\MMOTU\MMOTU\labels"
VAL_SPLIT = r"data\val.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MMOTUSegmentationDataset(
    IMAGE_DIR,
    MASK_DIR,
    VAL_SPLIT
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False
)

model = UNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        r"models\unet_mmotu.pth",
        map_location=DEVICE
    )
)

model.eval()


total_dice = 0.0
total_iou = 0.0
count = 0


with torch.no_grad():

    for images, masks in loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        predictions = (torch.sigmoid(outputs) > 0.5).float()

        intersection = (predictions * masks).sum(
            dim=(1, 2, 3)
        )

        prediction_area = predictions.sum(
            dim=(1, 2, 3)
        )

        mask_area = masks.sum(
            dim=(1, 2, 3)
        )

        union = prediction_area + mask_area - intersection

        dice = (
            (2 * intersection + 1e-6)
            / (prediction_area + mask_area + 1e-6)
        )

        iou = (
            (intersection + 1e-6)
            / (union + 1e-6)
        )

        total_dice += dice.sum().item()
        total_iou += iou.sum().item()

        count += images.size(0)


average_dice = total_dice / count
average_iou = total_iou / count


print("Evaluation complete.")
print(f"Validation images: {count}")
print(f"Average Dice: {average_dice:.4f}")
print(f"Average IoU: {average_iou:.4f}")