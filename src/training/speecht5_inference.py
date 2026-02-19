import copy
import json
import re
import time
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import SpeechT5ForTextToSpeech


@dataclass(frozen=True)
class DecodeProfile:
    name: str
    threshold: float
    minlenratio: float
    maxlenratio: float


BENCHMARK_FAST_PROFILE = DecodeProfile(
    name="benchmark_fast",
    threshold=0.50,
    minlenratio=0.0,
    maxlenratio=9.0,
)

ROBUST_FULL_PROFILE = DecodeProfile(
    name="robust_full",
    threshold=0.58,
    minlenratio=0.08,
    maxlenratio=18.0,
)

ROBUST_RETRY_PROFILE = DecodeProfile(
    name="robust_full_retry",
    threshold=0.65,
    minlenratio=0.12,
    maxlenratio=22.0,
)


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


def _normalize_text_for_tts(text, strong=False):
    normalized = str(text or "").strip()
    normalized = normalized.replace("\u2019", "'")
    normalized = normalized.replace("\u2018", "'")
    normalized = normalized.replace("\u201c", '"')
    normalized = normalized.replace("\u201d", '"')
    normalized = normalized.replace("\u2013", "-")
    normalized = normalized.replace("\u2014", "-")
    normalized = normalized.replace("...", ".")

    if strong:
        normalized = normalized.replace(",", " ")
        normalized = normalized.replace(";", " ")
        normalized = normalized.replace(":", " ")
        normalized = normalized.replace("(", " ")
        normalized = normalized.replace(")", " ")
    else:
        normalized = normalized.replace(";", ",")
        normalized = normalized.replace(":", ",")

    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _word_count(text):
    return len(re.findall(r"[A-Za-z']+", str(text or "")))


def _min_expected_duration_seconds(text, min_duration_per_word=0.24):
    words = max(1, _word_count(text))
    return max(1.0, words * float(min_duration_per_word))


def _generate_speech(
    model,
    processor,
    vocoder,
    text,
    speaker_embedding,
    device,
    profile=None,
    normalize_punctuation=True,
    return_meta=False,
):
    if profile is None:
        profile = ROBUST_FULL_PROFILE

    input_text = _normalize_text_for_tts(text) if normalize_punctuation else str(text or "")

    inputs = processor(text=input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    use_amp = hasattr(device, "type") and device.type == "cuda"
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            speech = model.generate_speech(
                input_ids,
                speaker_embedding,
                vocoder=vocoder,
                threshold=float(profile.threshold),
                minlenratio=float(profile.minlenratio),
                maxlenratio=float(profile.maxlenratio),
            )
    wav = speech.detach().float().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    if wav.ndim != 1:
        wav = wav.reshape(-1)
    if return_meta:
        return wav, {"text": input_text, "profile": profile.name}
    return wav


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


def _is_incomplete_generation(
    wav,
    text,
    sample_rate,
    min_duration_per_word=0.24,
):
    if wav is None:
        return True, "missing_wav"

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return True, "empty_wav"

    duration_sec = float(wav.size / float(sample_rate))
    expected = _min_expected_duration_seconds(text, min_duration_per_word=min_duration_per_word)
    if duration_sec < expected:
        return True, f"duration_too_short:{duration_sec:.2f}s<{expected:.2f}s"

    peak = float(np.max(np.abs(wav)))
    if peak <= 1e-8:
        return True, "silent_wav"

    end_window = max(1, int(sample_rate * 0.12))
    tail = np.abs(wav[-end_window:])
    tail_ratio = float(np.mean(tail > (peak * 0.015)))
    if tail_ratio < 0.01:
        return True, "dead_tail"

    return False, "ok"


def synthesize_test_sentences(
    model,
    processor,
    vocoder,
    speaker_embedding,
    sentences,
    output_dir,
    device,
    sample_rate=16000,
    profile=None,
    retry_profile=None,
    retry_for_completeness=True,
    normalize_punctuation=True,
    min_duration_per_word=0.24,
    trim_trailing_silence=True,
    silence_trim_db=-42.0,
    silence_keep_tail_ms=120.0,
):
    if profile is None:
        profile = ROBUST_FULL_PROFILE
    if retry_profile is None:
        retry_profile = ROBUST_RETRY_PROFILE

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spk = _to_speaker_tensor(speaker_embedding, device)
    output_paths = []

    for idx, text in enumerate(sentences, start=1):
        strong_text = _normalize_text_for_tts(text, strong=True)

        wav = _generate_speech(
            model,
            processor,
            vocoder,
            text,
            spk,
            device,
            profile=profile,
            normalize_punctuation=normalize_punctuation,
        )
        if trim_trailing_silence:
            wav = _trim_trailing_silence(
                wav,
                sample_rate=sample_rate,
                rel_db=silence_trim_db,
                keep_tail_ms=silence_keep_tail_ms,
            )

        incomplete, reason = _is_incomplete_generation(
            wav,
            text,
            sample_rate=sample_rate,
            min_duration_per_word=min_duration_per_word,
        )

        if retry_for_completeness and incomplete:
            wav = _generate_speech(
                model,
                processor,
                vocoder,
                text,
                spk,
                device,
                profile=retry_profile,
                normalize_punctuation=normalize_punctuation,
            )
            if trim_trailing_silence:
                wav = _trim_trailing_silence(
                    wav,
                    sample_rate=sample_rate,
                    rel_db=silence_trim_db,
                    keep_tail_ms=silence_keep_tail_ms,
                )
            incomplete, reason = _is_incomplete_generation(
                wav,
                text,
                sample_rate=sample_rate,
                min_duration_per_word=min_duration_per_word,
            )

        if retry_for_completeness and incomplete:
            wav = _generate_speech(
                model,
                processor,
                vocoder,
                strong_text,
                spk,
                device,
                profile=retry_profile,
                normalize_punctuation=False,
            )
            if trim_trailing_silence:
                wav = _trim_trailing_silence(
                    wav,
                    sample_rate=sample_rate,
                    rel_db=silence_trim_db,
                    keep_tail_ms=silence_keep_tail_ms,
                )
            incomplete, reason = _is_incomplete_generation(
                wav,
                text,
                sample_rate=sample_rate,
                min_duration_per_word=min_duration_per_word,
            )

        if incomplete:
            print(f"Warning: sample_{idx:02d} may be incomplete ({reason})")

        out_path = out_dir / f"sentence_{idx:02d}.wav"
        sf.write(str(out_path), wav, sample_rate)
        output_paths.append(out_path)

    return output_paths


def measure_latency(
    model,
    processor,
    vocoder,
    speaker_embedding,
    sentence,
    device,
    profile=None,
    warmup_runs=3,
    num_runs=20,
    normalize_punctuation=True,
):
    if profile is None:
        profile = BENCHMARK_FAST_PROFILE
    if sentence is None or str(sentence).strip() == "":
        return [], float("nan"), float("nan"), float("nan")

    spk = _to_speaker_tensor(speaker_embedding, device)
    text = str(sentence)

    for _ in range(max(0, int(warmup_runs))):
        _ = _generate_speech(
            model,
            processor,
            vocoder,
            text,
            spk,
            device,
            profile=profile,
            normalize_punctuation=normalize_punctuation,
        )

    latencies_ms = []
    for _ in range(max(1, int(num_runs))):
        _sync_cuda(device)
        t0 = time.perf_counter()
        _ = _generate_speech(
            model,
            processor,
            vocoder,
            text,
            spk,
            device,
            profile=profile,
            normalize_punctuation=normalize_punctuation,
        )
        _sync_cuda(device)
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(latencies_ms)) if latencies_ms else float("nan")
    p50_ms = float(np.percentile(latencies_ms, 50)) if latencies_ms else float("nan")
    p95_ms = float(np.percentile(latencies_ms, 95)) if latencies_ms else float("nan")
    return latencies_ms, mean_ms, p50_ms, p95_ms


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


def export_int8_deployment_bundle(
    model,
    processor,
    output_dir,
    base_model_id="microsoft/speecht5_tts",
    source_checkpoint=None,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    int8_path = export_int8_dynamic_artifact(model, out_dir / "model_int8_state_dict.pt")

    model.config.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.save_pretrained(str(out_dir))

    manifest = {
        "format": "speecht5_int8_dynamic_state_dict",
        "base_model_id": str(base_model_id),
        "source_checkpoint": str(source_checkpoint) if source_checkpoint is not None else None,
        "weights_file": str(int8_path.name),
        "total_size_mb": round(get_directory_size_mb(out_dir), 2),
    }
    (out_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return out_dir


def export_int8_dynamic_artifact(model, output_path):
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_model = copy.deepcopy(model).cpu().eval()
    qmodel = torch.quantization.quantize_dynamic(
        base_model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    torch.save(qmodel.state_dict(), str(out_path))
    return out_path


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
