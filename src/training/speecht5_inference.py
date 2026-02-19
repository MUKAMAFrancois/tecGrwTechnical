import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import SpeechT5ForTextToSpeech


def load_finetuned_model(checkpoint_path, device):
    # Avoid noisy warning during load if generation fields exist in model config.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Moving the following attributes in the config to the generation config:.*",
            category=UserWarning,
        )
        model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint_path).to(device)
    model.eval()
    return model


def configure_generation_for_latency(model, max_length=520):
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = int(max_length)
        if hasattr(model.generation_config, "num_beams"):
            model.generation_config.num_beams = 1
        if hasattr(model.generation_config, "do_sample"):
            model.generation_config.do_sample = False


def save_generation_config(model, checkpoint_path):
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.save_pretrained(checkpoint_path)


def _to_speaker_tensor(speaker_embedding, device):
    if torch.is_tensor(speaker_embedding):
        spk = speaker_embedding.to(device)
    else:
        spk = torch.tensor(speaker_embedding, dtype=torch.float32, device=device)
    if spk.dim() == 1:
        spk = spk.unsqueeze(0)
    return spk.float()


def _sync_cuda(device):
    if hasattr(device, "type") and device.type == "cuda":
        torch.cuda.synchronize(device)


def _generate_speech(model, processor, vocoder, text, speaker_embedding, device):
    inputs = processor(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    use_amp = hasattr(device, "type") and device.type == "cuda"
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            speech = model.generate_speech(input_ids, speaker_embedding, vocoder=vocoder)
    return speech.detach().float().cpu().numpy()


def synthesize_test_sentences(
    model,
    processor,
    vocoder,
    speaker_embedding,
    sentences,
    output_dir,
    device,
    sample_rate=16000,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spk = _to_speaker_tensor(speaker_embedding, device)
    output_paths = []

    for idx, text in enumerate(sentences, start=1):
        wav = _generate_speech(model, processor, vocoder, text, spk, device)
        out_path = out_dir / f"sentence_{idx:02d}.wav"
        sf.write(str(out_path), wav, sample_rate)
        output_paths.append(out_path)

    return output_paths


def measure_latency(
    model,
    processor,
    vocoder,
    speaker_embedding,
    sentences,
    device,
    warmup_runs=1,
):
    if not sentences:
        return [], float("nan")

    spk = _to_speaker_tensor(speaker_embedding, device)

    for _ in range(max(0, int(warmup_runs))):
        _ = _generate_speech(model, processor, vocoder, sentences[0], spk, device)

    latencies_ms = []
    for text in sentences:
        _sync_cuda(device)
        t0 = time.perf_counter()
        _ = _generate_speech(model, processor, vocoder, text, spk, device)
        _sync_cuda(device)
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(latencies_ms)) if latencies_ms else float("nan")
    return latencies_ms, mean_ms


def export_final_model_package(model, processor, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    processor.save_pretrained(str(out_dir))
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.save_pretrained(str(out_dir))
    return out_dir


def get_directory_size_mb(path):
    path_obj = Path(path)
    total_bytes = sum(p.stat().st_size for p in path_obj.rglob("*") if p.is_file())
    return total_bytes / (1024.0 * 1024.0)
