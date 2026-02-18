# src/model.py

import os
import torch
import torchaudio
from transformers import VitsModel, AutoTokenizer


def get_device():
    """
    Detect CUDA or CPU automatically.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(config):
    """
    Load MMS-TTS tokenizer.
    """

    model_name = config["MODEL_NAME"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer


def load_model(config, device=None):
    """
    Load pretrained MMS-TTS model.
    """

    if device is None:
        device = get_device()

    model_name = config["MODEL_NAME"]

    model = VitsModel.from_pretrained(model_name)

    model = model.to(device)

    model.eval()

    return model


def load_model_and_tokenizer(config, device=None):

    if device is None:
        device = get_device()

    tokenizer = load_tokenizer(config)

    model = load_model(config, device)

    return model, tokenizer


@torch.no_grad()
def synthesize_speech(
    model,
    tokenizer,
    text,
    device=None
):
    """
    Generate waveform from text.

    Returns:
        waveform tensor (CPU)
        sample_rate
    """

    if device is None:
        device = get_device()

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    output = model(**inputs)

    waveform = output.waveform

    waveform = waveform.cpu()

    sample_rate = model.config.sampling_rate

    return waveform, sample_rate


def save_waveform(
    waveform,
    sample_rate,
    path
):
    """
    Save waveform to WAV file.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    torchaudio.save(
        path,
        waveform,
        sample_rate
    )


def save_checkpoint(
    model,
    tokenizer,
    path
):
    """
    Save model and tokenizer checkpoint.
    """

    os.makedirs(path, exist_ok=True)

    model.save_pretrained(path)

    tokenizer.save_pretrained(path)

    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path,
    device=None
):
    """
    Load a fine-tuned checkpoint (standalone VitsModel, no LoRA).
    """

    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = VitsModel.from_pretrained(path)

    model = model.to(device)

    model.eval()

    return model, tokenizer


def freeze_model(model):

    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):

    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model):

    total = sum(p.numel() for p in model.parameters())

    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    return {
        "total": total,
        "trainable": trainable
    }
def set_train_mode(model):
    """
    Put model in training mode.
    """
    model.train()


def set_eval_mode(model):
    """
    Put model in evaluation mode.
    """
    model.eval()
