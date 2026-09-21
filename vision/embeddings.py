"""Vision Transformer embedding extraction."""

from __future__ import annotations


B2_MODEL_NAME = "google/vit-base-patch16-224-in21k"


def image_array_to_pil_rgb(image_array):
    import numpy as np
    from PIL import Image

    image = Image.fromarray(image_array.astype(np.uint8), mode="L")
    return image.convert("RGB")


def extract_vit_embeddings(cleaned_images, batch_size=4):
    import numpy as np
    import torch
    from sklearn.preprocessing import normalize
    from tqdm.auto import tqdm
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(B2_MODEL_NAME)
    model = AutoModel.from_pretrained(B2_MODEL_NAME).to(device)
    model.eval()

    batches = []
    with torch.no_grad():
        for start in tqdm(range(0, len(cleaned_images), batch_size), desc="ViT embeddings"):
            batch = cleaned_images[start:start + batch_size]
            images = [image_array_to_pil_rgb(image) for image in batch]
            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            batches.append(cls_embeddings)

    embeddings = np.vstack(batches)
    return normalize(embeddings)

