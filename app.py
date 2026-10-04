import streamlit as st
import torch
import numpy as np
from PIL import Image

from models.unet import UNet


# -----------------------------
# Settings
# -----------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = r"models\unet_mmotu.pth"


# -----------------------------
# Load model
# -----------------------------

@st.cache_resource
def load_model():

    model = UNet().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    return model


model = load_model()


# -----------------------------
# Page
# -----------------------------

st.set_page_config(
    page_title="Ovarian Tumor Segmentation",
    page_icon="🔬",
    layout="wide"
)

st.title("Ovarian Ultrasound Tumor Segmentation")

st.write(
    "Research prototype for segmenting tumor regions "
    "in ovarian ultrasound images."
)

st.info(
    "This is a research/educational model and is not a "
    "medical diagnostic tool."
)


# -----------------------------
# Upload image
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload an ultrasound image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Ultrasound")

    st.image(
        image,
        width="stretch"
    )


    # -----------------------------
    # Preprocess
    # -----------------------------

    resized_image = image.resize((256, 256))

    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    ) / 255.0

    image_tensor = torch.from_numpy(
        image_array
    ).permute(2, 0, 1).unsqueeze(0)


    # -----------------------------
    # Prediction
    # -----------------------------

    with torch.no_grad():

        output = model(
            image_tensor.to(DEVICE)
        )

        probability = torch.sigmoid(output)

        prediction = (
            probability > 0.5
        ).float()


    prediction = prediction.squeeze().cpu().numpy()


    # -----------------------------
    # Display result
    # -----------------------------

    st.subheader("Predicted Segmentation")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            prediction,
            caption="Predicted tumor region",
            clamp=True
        )

    with col2:

        overlay = image_array.copy()

        overlay[prediction > 0.5] = [
            1.0,
            0.0,
            0.0
        ]

        st.image(
            overlay,
            caption="Prediction overlay",
            clamp=True
        )