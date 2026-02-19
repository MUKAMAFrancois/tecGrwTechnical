import argparse
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


def _sanitize_wav_for_export(wav):
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(arr)))
    if peak > 1.0:
        arr = arr / peak
    arr = np.clip(arr, -1.0, 1.0)
    return arr.astype(np.float32, copy=False)


def load_int8_model(package_dir):
    pkg = Path(package_dir)
    state_path = pkg / "model_int8.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"INT8 weights not found: {state_path}")

    config = SpeechT5Config.from_pretrained(str(pkg), local_files_only=True)
    base_model = SpeechT5ForTextToSpeech(config)
    qmodel = torch.quantization.quantize_dynamic(
        base_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    state = torch.load(str(state_path), map_location="cpu")
    qmodel.load_state_dict(state)
    qmodel.cpu().eval()
    return qmodel


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
    wav = _sanitize_wav_for_export(wav)
    sf.write(str(out_path), wav, 16000, subtype="PCM_16")

    return {"path": out_path, "latency": latency, "device": device}


def run_int8(text, int8_dir, spk_emb, out_prefix):
    device = torch.device("cpu")
    _configure_runtime(device)
    print(f"\n[INT8] Loading model on {device} from: {int8_dir}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        processor = SpeechT5Processor.from_pretrained(str(int8_dir), local_files_only=True)
        model = load_int8_model(int8_dir)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device).eval()

    inputs = prepare_inputs(text, processor, device)
    _ = synthesize(model, inputs, spk_emb, vocoder, device)
    wav, latency = synthesize(model, inputs, spk_emb, vocoder, device)

    out_path = Path(f"{out_prefix}_int8.wav")
    wav = _sanitize_wav_for_export(wav)
    sf.write(str(out_path), wav, 16000, subtype="PCM_16")

    return {"path": out_path, "latency": latency, "device": device}


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

    if args.mode in ("fp32", "both"):
        fp32_dir = _resolve_dir(args.fp32_dir)
        results["FP32"] = run_fp32(args.text, fp32_dir, spk_emb, str(out_prefix))
        r = results["FP32"]
        print(f"[FP32] Saved: {r['path']} | Latency: {r['latency']:.2f} ms | Device: {r['device']}")

    if args.mode in ("int8", "both"):
        int8_dir = _resolve_dir(args.int8_dir)
        results["INT8"] = run_int8(args.text, int8_dir, spk_emb, str(out_prefix))
        r = results["INT8"]
        print(f"[INT8] Saved: {r['path']} | Latency: {r['latency']:.2f} ms | Device: {r['device']}")

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
