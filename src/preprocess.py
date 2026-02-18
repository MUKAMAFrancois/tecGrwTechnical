# src/preprocess.py

import os
import re
import csv
import torch
import torchaudio
import numpy as np
from tqdm import tqdm

from src.loader import (
    load_all_splits,
    ensure_dir,
    extract_sample_fields,
)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_valid_duration(duration_sec, min_sec=1.0, max_sec=15.0):
    return min_sec <= duration_sec <= max_sec


def trim_silence(audio_tensor, sample_rate):

    # torchaudio VAD expects CPU tensor
    audio_tensor = audio_tensor.cpu()

    trimmed = torchaudio.functional.vad(
        audio_tensor,
        sample_rate=sample_rate
    )

    # fallback if trimming removes everything
    if trimmed.numel() == 0:
        return audio_tensor

    return trimmed


def resample_audio(audio_tensor, orig_sr, target_sr):

    if orig_sr == target_sr:
        return audio_tensor

    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=target_sr
    )

    return resampler(audio_tensor)


def remove_dc_offset(audio_tensor):
    return audio_tensor - audio_tensor.mean()


def highpass_filter(audio_tensor, sample_rate, cutoff_hz=80):
    """Simple first-order high-pass to remove low-frequency rumble."""
    return torchaudio.functional.highpass_biquad(
        audio_tensor, sample_rate, cutoff_freq=cutoff_hz
    )


def normalize_audio(audio_tensor, target_db=-20.0):

    rms = torch.sqrt(torch.mean(audio_tensor**2))

    if rms == 0:
        return audio_tensor

    scalar = 10 ** (target_db / 20) / rms

    return audio_tensor * scalar


# Kinyarwanda digit words
_KIN_DIGITS = {
    "0": "zeru", "1": "rimwe", "2": "kabiri", "3": "gatatu",
    "4": "kane", "5": "gatanu", "6": "gatandatu",
    "7": "karindwi", "8": "umunani", "9": "icyenda",
}


def _expand_number(match):
    """Expand a number string digit-by-digit."""
    return " ".join(_KIN_DIGITS.get(d, d) for d in match.group())


_MACRON_MAP = {
    "ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u",
    "Ā": "A", "Ē": "E", "Ī": "I", "Ō": "O", "Ū": "U",
}


def _normalize_macrons(text):
    """Replace macron vowels with base vowels (ā→a, ē→e, etc.)."""
    for macron, base in _MACRON_MAP.items():
        text = text.replace(macron, base)
    return text


def clean_text(text):

    if text is None:
        return ""

    text = text.strip()

    # Normalize macron vowels to base vowels
    text = _normalize_macrons(text)

    # Expand digits to Kinyarwanda words
    text = re.sub(r"\d+", _expand_number, text)

    # Remove non-speech characters (keep letters, spaces, apostrophes, hyphens)
    text = re.sub(r"[^\w\s'\-']", "", text)

    text = " ".join(text.split())

    return text


def save_audio(audio_tensor, sample_rate, path):

    audio_tensor = audio_tensor.cpu()

    torchaudio.save(
        path,
        audio_tensor,
        sample_rate
    )


def process_sample(sample, config):

    try:
        target_sr = config["TARGET_SAMPLE_RATE"]

        fields = extract_sample_fields(sample)

        audio = torch.tensor(fields["audio_array"])

        if audio.ndim == 2:
            audio = audio.mean(dim=0)

        audio = audio.unsqueeze(0).float()

        sr = fields["sampling_rate"]
        text = clean_text(fields["text"])
        speaker_id = fields["speaker_id"]

        if not text:
            return None

        audio = resample_audio(audio, sr, target_sr)
        audio = remove_dc_offset(audio)
        audio = highpass_filter(audio, target_sr)
        audio = trim_silence(audio, target_sr)
        audio = normalize_audio(audio)

        audio = torch.clamp(audio, -1.0, 1.0)

        duration = audio.shape[1] / target_sr

        if not is_valid_duration(duration):
            return None

        return {
            "audio": audio,
            "text": text,
            "speaker_id": speaker_id,
            "duration": duration
        }

    except Exception as e:
        print(f"[Preprocess] Skipping sample due to error: {e}")
        return None



def process_split(split_name, dataset, config):

    output_dir = os.path.join(
        config["PROCESSED_DIR"],
        split_name
    )

    ensure_dir(output_dir)

    metadata = []

    selected_speaker = config["SELECTED_SPEAKER_ID"]

    counter = 0
    skipped =0

    for sample in tqdm(dataset, desc=f"Processing {split_name}"):

        if sample["speaker_id"] != selected_speaker:
            continue

        result = process_sample(sample, config)

        if result is None:
            skipped += 1
            continue

        filename = f"{split_name}_{counter:06d}.wav"

        path = os.path.join(output_dir, filename)

        save_audio(
            result["audio"],
            config["TARGET_SAMPLE_RATE"],
            path
        )

        metadata.append([
            path,
            result["text"],
            result["speaker_id"]
        ])

        counter += 1
        

    print(f"Preprocessing {split_name}: {counter} samples, {skipped} skipped")

    return metadata


def save_metadata(metadata, split_name, config):

    path = os.path.join(
        config["PROCESSED_DIR"],
        f"metadata_{split_name}.csv"
    )

    with open(path, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f, delimiter="|")

        for row in metadata:
            writer.writerow(row)


def run_preprocessing_pipeline(config, token=None):

    device = get_device()

    print(f"Using device: {device}")

    splits = load_all_splits(config, token)

    ensure_dir(config["PROCESSED_DIR"])

    stats = {}

    for split_name, dataset in splits.items():

        metadata = process_split(
            split_name,
            dataset,
            config
        )

        save_metadata(
            metadata,
            split_name,
            config
        )

        stats[split_name] = len(metadata)

    print("\nPreprocessing complete.")

    for split, count in stats.items():
        print(f"{split}: {count} samples")

    return stats
