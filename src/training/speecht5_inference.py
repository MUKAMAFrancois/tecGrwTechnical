import time
import warnings
import zipfile
import copy
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import SpeechT5Config, SpeechT5ForTextToSpeech

from src.preprocess import clean_text

INT8_META_FILE = "int8_quantization.json"


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


def configure_runtime_for_latency(device):
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if hasattr(device, "type") and device.type == "cuda":
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    else:
        cpu_threads = max(1, (os.cpu_count() or 2) - 1)
        torch.set_num_threads(cpu_threads)


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


def _prepare_generation_text(text, add_leading_prompt=True):
    cleaned = clean_text(text)
    if add_leading_prompt and cleaned:
        return "uuu " + cleaned
    return cleaned


def _prepare_input_ids(processor, text, device, add_leading_prompt=True):
    prepped_text = _prepare_generation_text(text, add_leading_prompt=add_leading_prompt)
    inputs = processor(text=prepped_text, return_tensors="pt")
    return inputs["input_ids"].to(device)


def _generate_speech_from_input_ids(
    model,
    vocoder,
    input_ids,
    speaker_embedding,
    device,
    threshold=0.5,
    minlenratio=0.0,
    maxlenratio=12.0,
):
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
    add_leading_prompt=True,
):
    input_ids = _prepare_input_ids(
        processor=processor,
        text=text,
        device=device,
        add_leading_prompt=add_leading_prompt,
    )
    return _generate_speech_from_input_ids(
        model=model,
        vocoder=vocoder,
        input_ids=input_ids,
        speaker_embedding=speaker_embedding,
        device=device,
        threshold=threshold,
        minlenratio=minlenratio,
        maxlenratio=maxlenratio,
    )


def _min_expected_duration_seconds(text):
    words = max(1, len(str(text).split()))
    return max(1.0, words * 0.24)


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


def _sanitize_wav_for_export(wav, target_peak=0.85, min_peak=1e-4, max_gain=12.0):
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(arr)))
    if peak <= 0.0:
        return arr.astype(np.float32, copy=False)
    if peak > 1.0:
        arr = arr / peak
        peak = 1.0
    if min_peak < peak < target_peak:
        gain = min(target_peak / peak, max_gain)
        arr = arr * gain
    arr = np.clip(arr, -1.0, 1.0)
    return arr.astype(np.float32, copy=False)


def _int8_qconfig_spec(model, scheme):
    normalized = str(scheme).strip().lower()
    if normalized == "all_linear":
        return {torch.nn.Linear}
    if normalized == "attention_only":
        linear_names = [
            name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
        ]
        selected = {
            name
            for name in linear_names
            if (".attention." in name) or (".self_attn." in name) or (".encoder_attn." in name)
        }
        if not selected:
            raise RuntimeError("No attention linear modules found for INT8 quantization.")
        return selected
    raise ValueError(f"Unsupported INT8 quantization scheme: {scheme}")


def _quantize_dynamic_speecht5(model, scheme):
    qconfig_spec = _int8_qconfig_spec(model, scheme)
    qmodel = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec=qconfig_spec,
        dtype=torch.qint8,
    )
    return qmodel.cpu().eval()


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
        wav = _sanitize_wav_for_export(wav)
        # PCM_16 improves compatibility with modern Windows Media Player.
        sf.write(str(out_path), wav, sample_rate, subtype="PCM_16")
        output_paths.append(out_path)

    return output_paths


def measure_latency(
    model,
    processor,
    vocoder,
    speaker_embedding,
    sentences,
    device,
    warmup_runs=2,
    maxlenratio=None,
    threshold=None,
    minlenratio=None,
    add_leading_prompt=False,
    cache_inputs=True,
):
    if not sentences:
        return [], float("nan")

    maxlenratio = float(6.8 if maxlenratio is None else maxlenratio)
    threshold = float(0.58 if threshold is None else threshold)
    minlenratio = float(0.0 if minlenratio is None else minlenratio)

    configure_runtime_for_latency(device)
    spk = _to_speaker_tensor(speaker_embedding, device)
    prepared_ids = None
    if cache_inputs:
        prepared_ids = [
            _prepare_input_ids(
                processor=processor,
                text=text,
                device=device,
                add_leading_prompt=add_leading_prompt,
            )
            for text in sentences
        ]

    for _ in range(max(0, int(warmup_runs))):
        if prepared_ids is not None:
            _ = _generate_speech_from_input_ids(
                model=model,
                vocoder=vocoder,
                input_ids=prepared_ids[0],
                speaker_embedding=spk,
                device=device,
                threshold=threshold,
                minlenratio=minlenratio,
                maxlenratio=maxlenratio,
            )
        else:
            _ = _generate_speech(
                model,
                processor,
                vocoder,
                sentences[0],
                spk,
                device,
                threshold=threshold,
                minlenratio=minlenratio,
                maxlenratio=maxlenratio,
                add_leading_prompt=add_leading_prompt,
            )

    latencies_ms = []
    for idx, text in enumerate(sentences):
        _sync_cuda(device)
        t0 = time.perf_counter()
        if prepared_ids is not None:
            _ = _generate_speech_from_input_ids(
                model=model,
                vocoder=vocoder,
                input_ids=prepared_ids[idx],
                speaker_embedding=spk,
                device=device,
                threshold=threshold,
                minlenratio=minlenratio,
                maxlenratio=maxlenratio,
            )
        else:
            _ = _generate_speech(
                model,
                processor,
                vocoder,
                text,
                spk,
                device,
                threshold=threshold,
                minlenratio=minlenratio,
                maxlenratio=maxlenratio,
                add_leading_prompt=add_leading_prompt,
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

    # Quantize and save INT8 state dict.
    # Attention-only quantization preserves intelligibility better for SpeechT5.
    scheme = "attention_only"
    base_model = copy.deepcopy(model).cpu().eval()
    qmodel = _quantize_dynamic_speecht5(base_model, scheme)
    torch.save(qmodel.state_dict(), str(out_dir / "model_int8.pt"))
    (out_dir / INT8_META_FILE).write_text(
        json.dumps({"scheme": scheme}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return out_dir


def load_int8_model(package_dir, device="cpu"):
    """Load an INT8 deployment package for inference."""
    pkg = Path(package_dir)
    state_path = pkg / "model_int8.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"INT8 weights not found: {state_path}")

    state = torch.load(str(state_path), map_location="cpu")
    schemes = ["attention_only", "all_linear"]
    meta_path = pkg / INT8_META_FILE
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            declared = str(meta.get("scheme", "")).strip().lower()
            if declared in schemes:
                schemes = [declared] + [s for s in schemes if s != declared]
        except Exception:
            pass

    errors = []
    for scheme in schemes:
        try:
            config = SpeechT5Config.from_pretrained(str(pkg), local_files_only=True)
            base_model = SpeechT5ForTextToSpeech(config)
            qmodel = _quantize_dynamic_speecht5(base_model, scheme)
            qmodel.load_state_dict(state)
            qmodel.to(device).eval()
            return qmodel
        except Exception as exc:
            errors.append(f"{scheme}: {exc}")

    raise RuntimeError("Unable to load INT8 model package. " + " | ".join(errors))


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
