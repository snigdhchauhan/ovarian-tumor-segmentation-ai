import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocessing.dataset import MMOTUSegmentationDataset
from models.unet import UNet


IMAGE_DIR = r"data\MMOTU\MMOTU\images"
MASK_DIR = r"data\MMOTU\MMOTU\labels"

TRAIN_SPLIT = r"data\train.txt"
VAL_SPLIT = r"data\val.txt"

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

train_dataset = MMOTUSegmentationDataset(
    IMAGE_DIR,
    MASK_DIR,
    TRAIN_SPLIT
)

val_dataset = MMOTUSegmentationDataset(
    IMAGE_DIR,
    MASK_DIR,
    VAL_SPLIT
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = UNet().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


def dice_score(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()

    intersection = (pred * target).sum()

    dice = (
        (2 * intersection + 1e-6)
        / (pred.sum() + target.sum() + 1e-6)
    )

    return dice.item()


for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()

    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, masks)

            val_loss += loss.item()
            val_dice += dice_score(outputs, masks)

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Dice: {val_dice:.4f}"
    )


torch.save(
    model.state_dict(),
    r"models\unet_mmotu.pth"
)

print("Training complete.")
print("Model saved to: models\\unet_mmotu.pth")