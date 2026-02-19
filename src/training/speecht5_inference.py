import copy
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


FAST_BENCHMARK_PROFILE = DecodeProfile(
    name="fast_benchmark",
    threshold=0.55,
    minlenratio=0.0,
    maxlenratio=8.0,
)

ROBUST_SYNTH_PROFILE = DecodeProfile(
    name="robust_synth",
    threshold=0.48,
    minlenratio=0.10,
    maxlenratio=14.0,
)

ROBUST_RETRY_PROFILE = DecodeProfile(
    name="robust_retry",
    threshold=0.42,
    minlenratio=0.14,
    maxlenratio=18.0,
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


def _normalize_inference_text(text, lowercase=False, strong=False):
    txt = str(text or "").strip()
    txt = txt.replace("\u2019", "'")
    txt = txt.replace("\u2018", "'")
    txt = txt.replace("\u201c", '"')
    txt = txt.replace("\u201d", '"')
    txt = txt.replace("\u2013", "-")
    txt = txt.replace("\u2014", "-")

    # Keep punctuation that helps prosody while removing odd symbols.
    txt = re.sub(r"[^\w\s\.\,\?\!'\-]", " ", txt)

    if strong:
        txt = txt.replace(",", " ")
        txt = txt.replace(";", " ")
        txt = txt.replace(":", " ")

    txt = re.sub(r"\s+", " ", txt).strip()
    if lowercase:
        txt = txt.lower()
    return txt


def _tokenize_text(processor, text, device):
    inputs = processor(text=text, return_tensors="pt")
    return inputs["input_ids"].to(device)


def _generate_speech(
    model,
    processor,
    vocoder,
    text,
    speaker_embedding,
    device,
    profile=None,
    normalize_text=True,
    lowercase_text=False,
    strong_text_norm=False,
    use_amp=True,
    input_ids=None,
):
    if profile is None:
        profile = ROBUST_SYNTH_PROFILE

    if input_ids is None:
        source_text = _normalize_inference_text(
            text,
            lowercase=lowercase_text if normalize_text else False,
            strong=strong_text_norm if normalize_text else False,
        )
        input_ids = _tokenize_text(processor, source_text, device)

    amp_enabled = bool(use_amp and hasattr(device, "type") and device.type == "cuda")
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
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
    return wav


def _min_expected_duration_seconds(text, min_duration_per_word=0.24):
    words = max(1, len(re.findall(r"[A-Za-z']+", str(text or ""))))
    return max(1.0, words * float(min_duration_per_word))


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


def _prepend_leading_silence(wav, sample_rate, pad_ms=24.0):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return wav
    pad = max(0, int((float(pad_ms) / 1000.0) * float(sample_rate)))
    if pad <= 0:
        return wav
    return np.concatenate([np.zeros(pad, dtype=np.float32), wav], axis=0)


def _is_incomplete_generation(
    wav,
    text,
    sample_rate,
    min_duration_per_word=0.24,
):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return True, "empty_wav"

    duration_sec = float(wav.size / float(sample_rate))
    expected_sec = _min_expected_duration_seconds(text, min_duration_per_word=min_duration_per_word)
    if duration_sec < expected_sec:
        return True, f"too_short:{duration_sec:.2f}s<{expected_sec:.2f}s"

    peak = float(np.max(np.abs(wav)))
    if peak <= 1e-8:
        return True, "silent"

    # Detect long dead tail.
    tail_window = max(1, int(sample_rate * 0.14))
    tail = np.abs(wav[-tail_window:])
    if float(np.mean(tail > (peak * 0.015))) < 0.01:
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
    fast_maxlenratio=9.0,
    safe_maxlenratio=18.0,
    retry_for_completeness=True,
    min_duration_per_word=0.24,
    quality_use_amp=False,
    prepend_leading_silence_ms=24.0,
    trim_trailing_silence=True,
    silence_trim_db=-42.0,
    silence_keep_tail_ms=120.0,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spk = _to_speaker_tensor(speaker_embedding, device)
    output_paths = []
    pass1_profile = DecodeProfile(
        name="synth_pass1",
        threshold=0.48,
        minlenratio=0.10,
        maxlenratio=float(max(10.0, fast_maxlenratio)),
    )
    pass2_profile = DecodeProfile(
        name="synth_pass2",
        threshold=0.42,
        minlenratio=0.14,
        maxlenratio=float(max(14.0, safe_maxlenratio)),
    )
    pass3_profile = DecodeProfile(
        name="synth_pass3",
        threshold=0.40,
        minlenratio=0.16,
        maxlenratio=float(max(16.0, safe_maxlenratio)),
    )

    for idx, text in enumerate(sentences, start=1):
        wav = _generate_speech(
            model,
            processor,
            vocoder,
            text,
            spk,
            device,
            profile=pass1_profile,
            normalize_text=True,
            lowercase_text=False,
            strong_text_norm=False,
            use_amp=quality_use_amp,
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
                profile=pass2_profile,
                normalize_text=True,
                lowercase_text=True,
                strong_text_norm=False,
                use_amp=quality_use_amp,
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
                profile=pass3_profile,
                normalize_text=True,
                lowercase_text=True,
                strong_text_norm=True,
                use_amp=quality_use_amp,
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
            print(f"Warning: possible incomplete synthesis for sentence_{idx:02d} ({reason})")

        wav = _prepend_leading_silence(
            wav,
            sample_rate=sample_rate,
            pad_ms=prepend_leading_silence_ms,
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
    warmup_runs=3,
    maxlenratio=11.0,
    threshold=0.40,
    sentence=None,
    num_runs=20,
    return_percentiles=False,
):
    has_sentence = sentence is not None and str(sentence).strip() != ""
    if not has_sentence and not sentences:
        if return_percentiles:
            return [], float("nan"), float("nan"), float("nan")
        return [], float("nan")

    spk = _to_speaker_tensor(speaker_embedding, device)
    bench_profile = DecodeProfile(
        name="latency_profile",
        threshold=float(threshold),
        minlenratio=0.0,
        maxlenratio=float(maxlenratio),
    )

    if has_sentence:
        bench_text = _normalize_inference_text(sentence, lowercase=True, strong=False)
        cached_input_ids = _tokenize_text(processor, bench_text, device)
        run_texts = [bench_text for _ in range(max(1, int(num_runs)))]
        warmup_text = bench_text
    else:
        run_texts = [_normalize_inference_text(t, lowercase=True, strong=False) for t in sentences]
        warmup_text = run_texts[0]
        cached_input_ids = None

    for _ in range(max(0, int(warmup_runs))):
        warmup_ids = cached_input_ids if has_sentence else None
        _ = _generate_speech(
            model,
            processor,
            vocoder,
            warmup_text,
            spk,
            device,
            profile=bench_profile,
            normalize_text=not has_sentence,
            lowercase_text=True,
            strong_text_norm=False,
            use_amp=True,
            input_ids=warmup_ids,
        )

    latencies_ms = []
    for text in run_texts:
        input_ids = cached_input_ids if has_sentence else None
        _sync_cuda(device)
        t0 = time.perf_counter()
        _ = _generate_speech(
            model,
            processor,
            vocoder,
            text,
            spk,
            device,
            profile=bench_profile,
            normalize_text=not has_sentence,
            lowercase_text=True,
            strong_text_norm=False,
            use_amp=True,
            input_ids=input_ids,
        )
        _sync_cuda(device)
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(latencies_ms)) if latencies_ms else float("nan")
    if return_percentiles:
        p50_ms = float(np.percentile(latencies_ms, 50)) if latencies_ms else float("nan")
        p95_ms = float(np.percentile(latencies_ms, 95)) if latencies_ms else float("nan")
        return latencies_ms, mean_ms, p50_ms, p95_ms
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
