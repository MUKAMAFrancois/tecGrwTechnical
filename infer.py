import argparse
import json
import os
import re
import time
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import soundfile as sf
import torch
from transformers import (
    SpeechT5Config,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
)

INT8_META_FILE = "int8_quantization.json"


try:
    from src.preprocess import clean_text
except ImportError:
    def clean_text(text):
        return re.sub(r"[^\w\s'\-]", "", text.strip())


def _sync_cuda(device):
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def _configure_runtime(device):
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if getattr(device, "type", None) == "cuda":
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    else:
        cpu_threads = max(1, (os.cpu_count() or 2) - 1)
        torch.set_num_threads(cpu_threads)


def _resolve_dir(path_or_zip):
    path = Path(path_or_zip)
    if path.is_dir():
        return path
    if path.is_file() and path.suffix.lower() == ".zip":
        out = path.with_suffix("")
        if not out.exists():
            with zipfile.ZipFile(str(path), "r") as zf:
                zf.extractall(str(out))
        return out
    raise FileNotFoundError(f"Expected directory or .zip path, got: {path}")


def _load_speaker_embedding(path):
    spk = torch.load(path, map_location="cpu")
    if not torch.is_tensor(spk):
        spk = torch.tensor(spk, dtype=torch.float32)
    if spk.dim() == 1:
        spk = spk.unsqueeze(0)
    return spk.float()


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


def _trim_trailing_silence(wav, sample_rate=16000):
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr

    sr = int(sample_rate)
    frame = max(1, int(0.020 * sr))  # 20 ms window
    hop = max(1, int(0.010 * sr))    # 10 ms hop
    if arr.size <= frame:
        return arr

    rms = []
    for start in range(0, arr.size - frame + 1, hop):
        chunk = arr[start:start + frame]
        rms.append(float(np.sqrt(np.mean(chunk * chunk))))
    if not rms:
        return arr

    rms = np.asarray(rms, dtype=np.float32)
    smooth = np.convolve(rms, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    peak_rms = float(np.max(smooth))
    if peak_rms <= 1e-8:
        return arr

    # Use a conservative floor estimate so low-energy final phonemes are preserved.
    noise_floor = float(np.percentile(smooth, 5))
    threshold = max(noise_floor * 1.8, peak_rms * 0.03)
    voiced = np.where(smooth > threshold)[0]
    if voiced.size == 0:
        return arr

    last_voiced = int(voiced[-1])
    trailing_silence_sec = (len(smooth) - 1 - last_voiced) * hop / float(sr)
    if trailing_silence_sec < 0.30:
        return arr

    keep_tail = int(0.100 * sr)  # keep 100 ms to avoid abrupt cutoff
    end = min(int(arr.size), last_voiced * hop + frame + keep_tail)
    return arr[:end]


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


def _load_int8_model_from_package(package_dir):
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
            return qmodel, f"package_{scheme}"
        except Exception as exc:
            errors.append(f"{scheme}: {exc}")

    raise RuntimeError(
        "Unable to load INT8 package with supported schemes. " + " | ".join(errors)
    )


def _build_runtime_int8_from_fp32(fp32_dir, scheme="attention_only"):
    base_model = SpeechT5ForTextToSpeech.from_pretrained(
        str(fp32_dir),
        local_files_only=True,
    ).cpu().eval()
    return _quantize_dynamic_speecht5(base_model, scheme)


def load_int8_model(package_dir, fp32_dir=None, strategy="auto"):
    normalized = str(strategy).strip().lower()
    if normalized not in {"auto", "runtime", "package"}:
        normalized = "auto"

    runtime_error = None
    if normalized in {"auto", "runtime"} and fp32_dir is not None:
        try:
            return _build_runtime_int8_from_fp32(fp32_dir), "runtime_attention_only"
        except Exception as exc:
            runtime_error = exc
            if normalized == "runtime":
                raise

    try:
        return _load_int8_model_from_package(package_dir)
    except Exception as package_exc:
        if runtime_error is not None:
            raise RuntimeError(
                f"INT8 runtime build failed: {runtime_error} ; "
                f"INT8 package load failed: {package_exc}"
            )
        raise


def prepare_inputs(text, processor, device):
    text_cleaned = clean_text(text)
    return processor(text=text_cleaned, return_tensors="pt").to(device)


def synthesize(model, inputs, spk_emb, vocoder, device):
    spk = spk_emb.to(device)
    _sync_cuda(device)
    t0 = time.perf_counter()

    use_amp = getattr(device, "type", None) == "cuda"
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            speech = model.generate_speech(
                inputs["input_ids"],
                spk,
                vocoder=vocoder,
                threshold=0.58,
                minlenratio=0.0,
                maxlenratio=6.8,
            )

    _sync_cuda(device)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0
    wav = speech.detach().cpu().numpy().astype(np.float32).reshape(-1)
    return wav, latency_ms


def run_fp32(text, fp32_dir, spk_emb, out_prefix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _configure_runtime(device)
    print(f"\n[FP32] Loading model on {device} from: {fp32_dir}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        processor = SpeechT5Processor.from_pretrained(str(fp32_dir), local_files_only=True)
        model = SpeechT5ForTextToSpeech.from_pretrained(
            str(fp32_dir), local_files_only=True
        ).to(device).eval()
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device).eval()

    inputs = prepare_inputs(text, processor, device)
    _ = synthesize(model, inputs, spk_emb, vocoder, device)
    wav, latency = synthesize(model, inputs, spk_emb, vocoder, device)

    out_path = Path(f"{out_prefix}_fp32.wav")
    wav = _trim_trailing_silence(wav, sample_rate=16000)
    wav = _sanitize_wav_for_export(wav)
    sf.write(str(out_path), wav, 16000, subtype="PCM_16")

    return {"path": out_path, "latency": latency, "device": device}


def run_int8(
    text,
    int8_dir,
    spk_emb,
    out_prefix,
    fp32_dir=None,
    int8_strategy="auto",
):
    device = torch.device("cpu")
    _configure_runtime(device)
    print(f"\n[INT8] Loading model on {device} from: {int8_dir}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model, model_source = load_int8_model(
            int8_dir,
            fp32_dir=fp32_dir,
            strategy=int8_strategy,
        )
        processor_dir = fp32_dir if str(model_source).startswith("runtime_") and fp32_dir else int8_dir
        processor = SpeechT5Processor.from_pretrained(str(processor_dir), local_files_only=True)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device).eval()

    inputs = prepare_inputs(text, processor, device)
    _ = synthesize(model, inputs, spk_emb, vocoder, device)
    wav, latency = synthesize(model, inputs, spk_emb, vocoder, device)

    out_path = Path(f"{out_prefix}_int8.wav")
    wav = _trim_trailing_silence(wav, sample_rate=16000)
    wav = _sanitize_wav_for_export(wav)
    sf.write(str(out_path), wav, 16000, subtype="PCM_16")

    return {
        "path": out_path,
        "latency": latency,
        "device": device,
        "model_source": model_source,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare FP32 vs INT8 SpeechT5 inference latency.")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--speaker", default="speaker_embedding.pt", help="Path to speaker embedding .pt")
    parser.add_argument(
        "--fp32_dir",
        default="speecht5_fp32_infer",
        help="FP32 model dir or zip (contains model.safetensors + config/tokenizer files).",
    )
    parser.add_argument(
        "--int8_dir",
        default="speecht5_int8_deployment",
        help="INT8 package dir or zip (contains model_int8.pt + config/tokenizer files).",
    )
    parser.add_argument(
        "--int8_strategy",
        default="auto",
        choices=["auto", "runtime", "package"],
        help=(
            "auto: build INT8 from FP32 when FP32 dir is available, else use package; "
            "runtime: always build from FP32; package: always load model_int8.pt package."
        ),
    )
    parser.add_argument("--mode", default="both", choices=["fp32", "int8", "both"])
    parser.add_argument("--out", default="output", help="Output file prefix")
    args = parser.parse_args()

    if not Path(args.speaker).exists():
        raise FileNotFoundError(f"Speaker embedding not found: {args.speaker}")

    spk_emb = _load_speaker_embedding(args.speaker)
    out_dir = Path("wav_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / args.out

    results = {}

    fp32_dir = None
    if args.mode in ("fp32", "both") or args.int8_strategy in ("auto", "runtime"):
        try:
            fp32_dir = _resolve_dir(args.fp32_dir)
        except Exception:
            if args.mode in ("fp32", "both"):
                raise
            fp32_dir = None

    if args.mode in ("fp32", "both"):
        if fp32_dir is None:
            fp32_dir = _resolve_dir(args.fp32_dir)
        results["FP32"] = run_fp32(args.text, fp32_dir, spk_emb, str(out_prefix))
        r = results["FP32"]
        print(f"[FP32] Saved: {r['path']} | Latency: {r['latency']:.2f} ms | Device: {r['device']}")

    if args.mode in ("int8", "both"):
        int8_dir = _resolve_dir(args.int8_dir)
        results["INT8"] = run_int8(
            args.text,
            int8_dir,
            spk_emb,
            str(out_prefix),
            fp32_dir=fp32_dir,
            int8_strategy=args.int8_strategy,
        )
        r = results["INT8"]
        print(
            f"[INT8] Saved: {r['path']} | Latency: {r['latency']:.2f} ms | "
            f"Device: {r['device']} | Source: {r['model_source']}"
        )

    if args.mode == "both" and len(results) == 2:
        fp = results["FP32"]["latency"]
        q = results["INT8"]["latency"]
        ratio = (fp / q) if q > 0 else float("inf")
        print("\n==================== COMPARISON ====================")
        print(f"FP32 : {fp:.2f} ms ({results['FP32']['device']})")
        print(f"INT8 : {q:.2f} ms ({results['INT8']['device']})")
        print(f"FP32/INT8 latency ratio: {ratio:.3f}x")


if __name__ == "__main__":
    main()
