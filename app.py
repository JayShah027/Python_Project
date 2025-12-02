# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os
import zipfile
import gdown
from pathlib import Path

# Must be called early
st.set_page_config(page_title="Image Classifier (Streamlit Cloud)")

# Use a meaningful env var name. In Streamlit Cloud, set this under Secrets / Environment variables.
GDRIVE_FILE_ID = os.environ.get("1izp4Qrh-FU4pQ_33DyLe33GnsdP7R7dh", "")  # set MODEL_GDRIVE_ID in Streamlit settings
MODEL_DIR = Path("models/final_saved_model.keras")
LABELS_PATH = Path("models/labels.json")


@st.cache_resource
def ensure_model_available():
    global MODEL_DIR, LABELS_PATH   # <-- FIXED: declare globals first

    # already present?
    if MODEL_DIR.exists() and LABELS_PATH.exists():
        return True

    # no env var and not in repo => raise
    if not GDRIVE_FILE_ID:
        raise RuntimeError(
            "Model not found in repo and no MODEL_GDRIVE_ID set. "
            "Set MODEL_GDRIVE_ID in Streamlit Cloud or commit a models/ folder to the repo."
        )

    os.makedirs("models", exist_ok=True)
    zip_path = "models/model.zip"

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    st.info("Downloading model (this runs only once at startup)...")
    gdown.download(url, zip_path, quiet=False)

    # extract zip
    import zipfile
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("models")
    except zipfile.BadZipFile as e:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise RuntimeError(f"Downloaded file is not a valid zip: {e}")

    # cleanup
    try:
        os.remove(zip_path)
    except OSError:
        pass

    # -------------------------------------------------------------
    # AUTO-DETECT MODEL FILE OR FOLDER INSIDE models/
    # -------------------------------------------------------------
    if not MODEL_DIR.exists():
        MODEL_DIR_resolved = None

        for entry in Path("models").iterdir():
            if entry.is_dir():
                if (entry / "saved_model.pb").exists():
                    MODEL_DIR_resolved = entry
                    break
                if entry.name.endswith(".keras"):
                    MODEL_DIR_resolved = entry
                    break
            else:
                if entry.suffix in {".h5", ".keras"}:
                    MODEL_DIR_resolved = entry
                    break

        if MODEL_DIR_resolved is None:
            raise RuntimeError("Downloaded model is missing expected files.")

        MODEL_DIR = MODEL_DIR_resolved   # <-- update global

    # -------------------------------------------------------------
    # FIND LABELS.JSON ANYWHERE INSIDE models/
    # -------------------------------------------------------------
    if not LABELS_PATH.exists():
        found = list(Path("models").rglob("labels.json"))
        if not found:
            raise RuntimeError("labels.json not found inside model archive.")
        LABELS_PATH = found[0]   # <-- update global

    return True


@st.cache_resource
def load_model_and_labels():
    ensure_model_available()
    # tf.keras.models.load_model accepts a folder (SavedModel) or a file (.h5/.keras)
    model = tf.keras.models.load_model(str(MODEL_DIR))
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    return model, labels


def preprocess_image(img: Image.Image, size=(224, 224)):
    img = img.convert("RGB").resize(size)
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    return img, x


def main():
    st.title("Image Classifier Demo (Streamlit Cloud)")

    # load model
    try:
        with st.spinner("Loading model..."):
            model, labels = load_model_and_labels()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        # provide some hints
        st.caption("Tip: set MODEL_GDRIVE_ID in Streamlit Cloud or commit a models/ folder with final_saved_model.keras and labels.json")
        return

    uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if not uploaded:
        st.info("Upload an image to get a prediction.")
        return

    img, x = preprocess_image(Image.open(uploaded))
    st.image(img, caption="Uploaded image", use_column_width=True)

    # prediction
    try:
        preds = model.predict(x)[0]
        top_idx = int(np.argmax(preds))
        st.write(f"Prediction: **{labels[top_idx]}** ({preds[top_idx]*100:.2f}%)")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
