import time
import warnings
import zipfile
import copy
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import SpeechT5ForTextToSpeech

from src.preprocess import clean_text


def load_finetuned_model(checkpoint_path, device):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Moving the following attributes in the config to the generation config:.*",
            category=UserWarning,
        )
        model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint_path).to(device)
    model.eval()
    return model


def configure_generation_for_latency(model, max_length=600):
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


def _generate_speech(
    model,
    processor,
    vocoder,
    text,
    speaker_embedding,
    device,
    threshold=0.5,
    minlenratio=0.0,
    maxlenratio=12.0,
):
    text = "uuu " + clean_text(text)
    inputs = processor(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    use_amp = hasattr(device, "type") and device.type == "cuda"
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            speech = model.generate_speech(
                input_ids,
                speaker_embedding,
                vocoder=vocoder,
                threshold=threshold,
                minlenratio=minlenratio,
                maxlenratio=maxlenratio,
            )
    wav = speech.detach().float().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    if wav.ndim != 1:
        wav = wav.reshape(-1)
    return wav


def _min_expected_duration_seconds(text):
    words = max(1, len(str(text).split()))
    return max(2.0, words * 0.45)


def _trim_trailing_silence(
    wav,
    sample_rate,
    rel_db=-42.0,
    keep_tail_ms=120.0,
):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return wav

    peak = float(np.max(np.abs(wav)))
    if peak <= 1e-8:
        return wav

    amplitude_threshold = peak * float(10.0 ** (rel_db / 20.0))
    voiced = np.where(np.abs(wav) > amplitude_threshold)[0]
    if voiced.size == 0:
        return wav

    keep_tail = max(0, int((keep_tail_ms / 1000.0) * float(sample_rate)))
    end = min(int(wav.size), int(voiced[-1]) + keep_tail + 1)
    return wav[:end]


def synthesize_test_sentences(
    model,
    processor,
    vocoder,
    speaker_embedding,
    sentences,
    output_dir,
    device,
    sample_rate=16000,
    fast_maxlenratio=10.0,
    safe_maxlenratio=14.0,
    retry_for_completeness=True,
    trim_trailing_silence=True,
    silence_trim_db=-42.0,
    silence_keep_tail_ms=120.0,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spk = _to_speaker_tensor(speaker_embedding, device)
    output_paths = []

    for idx, text in enumerate(sentences, start=1):
        wav = _generate_speech(
            model,
            processor,
            vocoder,
            text,
            spk,
            device,
            threshold=0.50,
            maxlenratio=fast_maxlenratio,
        )
        duration_sec = float(len(wav) / float(sample_rate)) if len(wav) > 0 else 0.0
        if retry_for_completeness and duration_sec < _min_expected_duration_seconds(text):
            wav = _generate_speech(
                model,
                processor,
                vocoder,
                text,
                spk,
                device,
                threshold=0.50,
                maxlenratio=safe_maxlenratio,
            )

        if trim_trailing_silence:
            wav = _trim_trailing_silence(
                wav,
                sample_rate=sample_rate,
                rel_db=silence_trim_db,
                keep_tail_ms=silence_keep_tail_ms,
            )

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
    maxlenratio=11.0,
    threshold=0.40,
):
    if not sentences:
        return [], float("nan")

    spk = _to_speaker_tensor(speaker_embedding, device)

    for _ in range(max(0, int(warmup_runs))):
        _ = _generate_speech(
            model,
            processor,
            vocoder,
            sentences[0],
            spk,
            device,
            threshold=threshold,
            maxlenratio=maxlenratio,
        )

    latencies_ms = []
    for text in sentences:
        _sync_cuda(device)
        t0 = time.perf_counter()
        _ = _generate_speech(
            model,
            processor,
            vocoder,
            text,
            spk,
            device,
            threshold=threshold,
            maxlenratio=maxlenratio,
        )
        _sync_cuda(device)
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(latencies_ms)) if latencies_ms else float("nan")
    return latencies_ms, mean_ms


def export_final_model_package(model, processor, output_dir, save_dtype="float32"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    normalized = str(save_dtype).lower()
    if normalized in {"fp16", "float16", "half"}:
        state_dict = {
            key: (
                value.detach().cpu().to(torch.float16)
                if torch.is_tensor(value) and torch.is_floating_point(value)
                else value.detach().cpu()
                if torch.is_tensor(value)
                else value
            )
            for key, value in state_dict.items()
        }
    else:
        state_dict = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in state_dict.items()
        }

    model.save_pretrained(str(out_dir), safe_serialization=True, state_dict=state_dict)
    processor.save_pretrained(str(out_dir))
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.save_pretrained(str(out_dir))
    return out_dir


def export_int8_deployment_package(model, processor, output_dir):
    """Export a complete INT8 deployment package (model + processor + config)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save processor (tokenizer + feature extractor)
    processor.save_pretrained(str(out_dir))

    # Save model config
    model.config.save_pretrained(str(out_dir))

    # Save generation config
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.save_pretrained(str(out_dir))

    # Quantize and save INT8 state dict
    base_model = copy.deepcopy(model).cpu().eval()
    qmodel = torch.quantization.quantize_dynamic(
        base_model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    torch.save(qmodel.state_dict(), str(out_dir / "model_int8.pt"))
    return out_dir


def load_int8_model(package_dir, device="cpu"):
    """Load an INT8 deployment package for inference."""
    pkg = Path(package_dir)
    base_model = SpeechT5ForTextToSpeech.from_pretrained(str(pkg))
    qmodel = torch.quantization.quantize_dynamic(
        base_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    state = torch.load(str(pkg / "model_int8.pt"), map_location="cpu")
    qmodel.load_state_dict(state)
    qmodel.to(device).eval()
    return qmodel


def get_directory_size_mb(path):
    path_obj = Path(path)
    total_bytes = sum(p.stat().st_size for p in path_obj.rglob("*") if p.is_file())
    return total_bytes / (1024.0 * 1024.0)


def get_file_size_mb(path):
    return Path(path).stat().st_size / (1024.0 * 1024.0)


def zip_directory(path, zip_path):
    src = Path(path)
    dst = Path(zip_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(dst), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in src.rglob("*"):
            if file_path.is_file():
                zf.write(str(file_path), arcname=str(file_path.relative_to(src)))
    return dst
