import torch
import matplotlib.pyplot as plt

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

model = UNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        r"models\unet_mmotu.pth",
        map_location=DEVICE
    )
)

model.eval()


image, mask = dataset[0]

with torch.no_grad():
    output = model(
        image.unsqueeze(0).to(DEVICE)
    )

    prediction = (
        torch.sigmoid(output) > 0.5
    ).float()


image_np = image.permute(1, 2, 0).numpy()
mask_np = mask.squeeze().numpy()
prediction_np = prediction.squeeze().cpu().numpy()


plt.figure(figsize=(16, 4))

# Original ultrasound
plt.subplot(1, 4, 1)
plt.imshow(image_np)
plt.title("Original Ultrasound")
plt.axis("off")

# Ground truth
plt.subplot(1, 4, 2)
plt.imshow(mask_np, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

# Prediction
plt.subplot(1, 4, 3)
plt.imshow(prediction_np, cmap="gray")
plt.title("Model Prediction")
plt.axis("off")

# Overlay
plt.subplot(1, 4, 4)
plt.imshow(image_np)
plt.imshow(
    prediction_np,
    cmap="Reds",
    alpha=0.45
)
plt.title("Prediction Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()