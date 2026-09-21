"""TrOCR baseline inference and optional fine-tuning."""

from __future__ import annotations

import gc
import re
import unicodedata
from pathlib import Path

from ocr.crops import write_predictions_and_score


B3_BASELINE_MODEL_NAME = "microsoft/trocr-base-printed"
B3_FT_MODEL_NAME = "microsoft/trocr-small-printed"
B3_BATCH_SIZE = 1
B3_GRAD_ACCUM_STEPS = 16
B3_LR_DECODER = 2e-5
B3_LR_ENCODER = 1e-5


def crop_to_pil(gray_crop):
    import cv2
    from PIL import Image

    rgb = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def normalize_prediction(text):
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).upper()
    for src, tgt in {"0": "O", "1": "I", "|": "I", "5": "S", "8": "B"}.items():
        text = text.replace(src, tgt)
    return re.sub(r"[^A-Z]", "", text)


def trocr_predict(model, processor, line_crops, line_indices, device, batch_size=4, normalize=False):
    import torch
    from tqdm.auto import tqdm

    model.eval()
    preds = {}

    with torch.no_grad():
        for start in tqdm(range(0, len(line_indices), batch_size), desc="TrOCR inference"):
            batch_idx = line_indices[start:start + batch_size]
            images = [crop_to_pil(line_crops[index]) for index in batch_idx]
            pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values, max_new_tokens=64, num_beams=1)
            texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for index, text in zip(batch_idx, texts):
                preds[index] = normalize_prediction(text) if normalize else text
    return preds


def run_baseline(lines_df, line_crops, eval_root, gt_dir, batch_size):
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(B3_BASELINE_MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(B3_BASELINE_MODEL_NAME).to(device)

    baseline_test = lines_df[
        (lines_df["split"] == "test")
        & (lines_df["inscription_id"].str.endswith("_Rotation1_300dpi"))
    ].copy()
    line_indices = baseline_test["line_index"].tolist()
    print(f"Baseline will transcribe {len(line_indices)} test line crops.")

    raw_preds = trocr_predict(model, processor, line_crops, line_indices, device, batch_size=batch_size, normalize=False)
    raw_cer = write_predictions_and_score(lines_df, raw_preds, "pred_baseline_trocr_base", eval_root, gt_dir)
    clean_preds = {line_index: normalize_prediction(pred) for line_index, pred in raw_preds.items()}
    clean_cer = write_predictions_and_score(lines_df, clean_preds, "pred_baseline_trocr_base_cleaned", eval_root, gt_dir)

    model.to("cpu")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "processor": processor,
        "line_indices": line_indices,
        "raw_predictions": raw_preds,
        "clean_predictions": clean_preds,
        "raw_cer": raw_cer,
        "clean_cer": clean_cer,
        "device": device,
    }


def configure_trocr_for_training_and_generation(model, processor, max_len):
    tokenizer = processor.tokenizer
    decoder_start_id = tokenizer.cls_token_id
    eos_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    model.config.decoder_start_token_id = decoder_start_id
    model.config.eos_token_id = eos_id
    model.config.pad_token_id = pad_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.use_cache = False

    model.generation_config.decoder_start_token_id = decoder_start_id
    model.generation_config.eos_token_id = eos_id
    model.generation_config.pad_token_id = pad_id
    model.generation_config.max_length = max_len
    model.generation_config.num_beams = 1
    return model


def finetune_trocr(lines_df, line_crops, baseline_line_indices, eval_root, gt_dir, output_dir, epochs, batch_size):
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(B3_FT_MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(B3_FT_MODEL_NAME)

    max_target_len = min(128, max(64, int(lines_df["n_chars"].max()) + 4))
    model = configure_trocr_for_training_and_generation(model, processor, max_target_len)

    for param in model.encoder.parameters():
        param.requires_grad = False
    try:
        for layer in model.encoder.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
    except Exception as error:
        print(f"Could not unfreeze last two encoder layers: {error}")
    for param in model.decoder.parameters():
        param.requires_grad = True
    try:
        model.gradient_checkpointing_enable()
    except Exception as error:
        print(f"Gradient checkpointing could not be enabled: {error}")

    model.to(device)
    train_full_df = lines_df[lines_df["split"] == "train"].reset_index(drop=True)
    train_bases_all = sorted(train_full_df["base_id"].unique())
    if len(train_bases_all) < 2:
        raise ValueError("Fine-tuning needs at least two training base inscriptions.")

    ft_train_bases, ft_val_bases = train_test_split(train_bases_all, test_size=0.10, random_state=123)
    train_df = train_full_df[train_full_df["base_id"].isin(set(ft_train_bases))].reset_index(drop=True)
    val_df = train_full_df[train_full_df["base_id"].isin(set(ft_val_bases))].reset_index(drop=True)

    class SqueezeLineDataset(Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image = crop_to_pil(line_crops[row["line_index"]])
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
            labels = processor.tokenizer(
                row["text"],
                padding="max_length",
                max_length=max_target_len,
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            labels[labels == processor.tokenizer.pad_token_id] = -100
            return {"pixel_values": pixel_values, "labels": labels}

    train_loader = DataLoader(
        SqueezeLineDataset(train_df),
        batch_size=B3_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        SqueezeLineDataset(val_df),
        batch_size=B3_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    decoder_params = []
    encoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder"):
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    optimizer_groups = []
    if decoder_params:
        optimizer_groups.append({"params": decoder_params, "lr": B3_LR_DECODER})
    if encoder_params:
        optimizer_groups.append({"params": encoder_params, "lr": B3_LR_ENCODER})

    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    @torch.no_grad()
    def compute_val_loss(max_batches=200):
        model.eval()
        total_loss = 0.0
        total_batches = 0
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
            total_batches += 1
        model.train()
        return None if total_batches == 0 else total_loss / total_batches

    training_rows = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Fine-tuning epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(progress):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / B3_GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            should_step = (step + 1) % B3_GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_loss = loss.item() * B3_GRAD_ACCUM_STEPS
            running_loss += batch_loss
            progress.set_postfix(loss=f"{batch_loss:.3f}")

        mean_train_loss = running_loss / len(train_loader)
        mean_val_loss = compute_val_loss()
        training_rows.append({"epoch": epoch + 1, "train_loss": mean_train_loss, "val_loss": mean_val_loss})
        print(f"Epoch {epoch + 1}: train={mean_train_loss:.4f}, val={mean_val_loss}")

    model.eval()
    ft_preds = trocr_predict(
        model=model,
        processor=processor,
        line_crops=line_crops,
        line_indices=baseline_line_indices,
        device=device,
        batch_size=batch_size,
        normalize=True,
    )
    ft_cer = write_predictions_and_score(lines_df, ft_preds, "pred_trocr_small_finetuned", eval_root, gt_dir)

    model_dir = Path(output_dir) / "trocr-small-finetuned"
    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)
    pd.DataFrame(training_rows).to_csv(Path(output_dir) / "finetune_losses.csv", index=False)
    return ft_preds, ft_cer, model_dir

