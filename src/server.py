import io
import json
import os
import re
import time
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import (
    SpeechT5Config,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
)

INT8_META_FILE = "int8_quantization.json"

try:
    from src.preprocess import clean_text
except Exception:
    def clean_text(text):
        return re.sub(r"[^\w\s'\-]", "", str(text).strip())


class SynthesizeRequest(BaseModel):
    text: str
    mode: Literal["fp32", "int8"] = "fp32"


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


class TTSRuntime:
    def __init__(self):
        self.fp32_dir = Path(os.getenv("TTS_FP32_DIR", "/models/speecht5_fp32_infer"))
        self.int8_dir = Path(os.getenv("TTS_INT8_DIR", "/models/speecht5_int8_deployment"))
        self.speaker_path = Path(os.getenv("TTS_SPEAKER_PATH", "/models/speaker_embedding.pt"))
        self.vocoder_id = os.getenv("TTS_VOCODER_ID", "microsoft/speecht5_hifigan")
        self.sample_rate = int(os.getenv("TTS_SAMPLE_RATE", "16000"))
        self.device = self._resolve_device(os.getenv("TTS_DEVICE", "auto"))
        self.int8_strategy = str(os.getenv("TTS_INT8_STRATEGY", "auto")).strip().lower()
        if self.int8_strategy not in {"auto", "runtime", "package"}:
            self.int8_strategy = "auto"

        self._speaker_embedding = None
        self._vocoder_by_device = {}

        self._processor_fp32 = None
        self._model_fp32 = None

        self._processor_int8 = None
        self._model_int8 = None
        self._int8_source = None

        self._configure_runtime(self.device)

    def _resolve_device(self, raw_device):
        raw = str(raw_device).strip().lower()
        if raw == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(raw)

    def _configure_runtime(self, device):
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

    def _sync_cuda(self, device):
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)

    def _get_speaker_embedding(self):
        if self._speaker_embedding is not None:
            return self._speaker_embedding

        if not self.speaker_path.exists():
            raise FileNotFoundError(f"Speaker embedding not found: {self.speaker_path}")

        spk = torch.load(str(self.speaker_path), map_location="cpu")
        if not torch.is_tensor(spk):
            spk = torch.tensor(spk, dtype=torch.float32)
        if spk.dim() == 1:
            spk = spk.unsqueeze(0)
        self._speaker_embedding = spk.float()
        return self._speaker_embedding

    def _get_vocoder(self, device):
        key = str(device)
        if key in self._vocoder_by_device:
            return self._vocoder_by_device[key]

        vocoder = SpeechT5HifiGan.from_pretrained(self.vocoder_id).to(device).eval()
        self._vocoder_by_device[key] = vocoder
        return vocoder

    def _ensure_fp32_bundle(self):
        if self._processor_fp32 is not None and self._model_fp32 is not None:
            return self._processor_fp32, self._model_fp32

        if not self.fp32_dir.exists():
            raise FileNotFoundError(f"FP32 model directory not found: {self.fp32_dir}")

        self._processor_fp32 = SpeechT5Processor.from_pretrained(
            str(self.fp32_dir), local_files_only=True
        )
        self._model_fp32 = SpeechT5ForTextToSpeech.from_pretrained(
            str(self.fp32_dir), local_files_only=True
        ).to(self.device).eval()
        return self._processor_fp32, self._model_fp32

    def _load_int8_model(self):
        runtime_error = None
        if self.int8_strategy in {"auto", "runtime"} and self.fp32_dir.exists():
            try:
                base_model = SpeechT5ForTextToSpeech.from_pretrained(
                    str(self.fp32_dir), local_files_only=True
                ).cpu().eval()
                qmodel = _quantize_dynamic_speecht5(base_model, "attention_only")
                return qmodel, "runtime_attention_only"
            except Exception as exc:
                runtime_error = exc
                if self.int8_strategy == "runtime":
                    raise

        state_path = self.int8_dir / "model_int8.pt"
        if not state_path.exists():
            if runtime_error is not None:
                raise RuntimeError(
                    f"INT8 runtime build failed: {runtime_error} ; "
                    f"INT8 weights not found: {state_path}"
                )
            raise FileNotFoundError(f"INT8 weights not found: {state_path}")

        state = torch.load(str(state_path), map_location="cpu")
        schemes = ["attention_only", "all_linear"]
        meta_path = self.int8_dir / INT8_META_FILE
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
                config = SpeechT5Config.from_pretrained(str(self.int8_dir), local_files_only=True)
                base_model = SpeechT5ForTextToSpeech(config)
                qmodel = _quantize_dynamic_speecht5(base_model, scheme)
                qmodel.load_state_dict(state)
                return qmodel.cpu().eval(), f"package_{scheme}"
            except Exception as exc:
                errors.append(f"{scheme}: {exc}")

        if runtime_error is not None:
            errors.append(f"runtime: {runtime_error}")
        raise RuntimeError("Unable to load INT8 model. " + " | ".join(errors))

    def _ensure_int8_bundle(self):
        if self._processor_int8 is not None and self._model_int8 is not None:
            return self._processor_int8, self._model_int8

        if self.int8_strategy == "package" and not self.int8_dir.exists():
            raise FileNotFoundError(f"INT8 model directory not found: {self.int8_dir}")

        self._model_int8, self._int8_source = self._load_int8_model()
        processor_dir = (
            self.fp32_dir
            if str(self._int8_source).startswith("runtime_") and self.fp32_dir.exists()
            else self.int8_dir
        )
        self._processor_int8 = SpeechT5Processor.from_pretrained(
            str(processor_dir), local_files_only=True
        )
        return self._processor_int8, self._model_int8

    def synthesize(self, text, mode):
        if not str(text).strip():
            raise ValueError("Input text is empty.")

        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("Input text becomes empty after cleaning.")

        if mode == "fp32":
            processor, model = self._ensure_fp32_bundle()
            device = self.device
        elif mode == "int8":
            processor, model = self._ensure_int8_bundle()
            device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        vocoder = self._get_vocoder(device)
        speaker_embedding = self._get_speaker_embedding().to(device)
        inputs = processor(text=cleaned, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        self._sync_cuda(device)
        t0 = time.perf_counter()
        use_amp = getattr(device, "type", None) == "cuda"
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                speech = model.generate_speech(
                    input_ids,
                    speaker_embedding,
                    vocoder=vocoder,
                    threshold=0.58,
                    minlenratio=0.0,
                    maxlenratio=6.8,
                )
        self._sync_cuda(device)
        t1 = time.perf_counter()

        wav = speech.detach().cpu().numpy().astype(np.float32).reshape(-1)
        wav = _sanitize_wav_for_export(wav)
        return wav, self.sample_rate, (t1 - t0) * 1000.0

    def health(self):
        return {
            "status": "ok",
            "device": str(self.device),
            "fp32_dir_exists": self.fp32_dir.exists(),
            "int8_dir_exists": self.int8_dir.exists(),
            "speaker_exists": self.speaker_path.exists(),
            "int8_strategy": self.int8_strategy,
            "int8_source": self._int8_source,
        }


app = FastAPI(title="TechnicalTTS API", version="1.0.0")
runtime = TTSRuntime()


@app.get("/health")
def health():
    return JSONResponse(runtime.health())


@app.post(
    "/synthesize",
    response_class=Response,
    responses={
        200: {
            "description": "Synthesized speech WAV audio.",
            "content": {
                "audio/wav": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        }
    },
)
def synthesize(request: SynthesizeRequest):
    try:
        wav, sr, latency_ms = runtime.synthesize(request.text, request.mode)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    buffer = io.BytesIO()
    sf.write(buffer, wav, sr, format="WAV", subtype="PCM_16")
    buffer.seek(0)

    headers = {
        "X-TTS-Mode": request.mode,
        "X-TTS-Latency-Ms": f"{latency_ms:.2f}",
        "Content-Disposition": f'attachment; filename="synthesize_{request.mode}.wav"',
    }
    return Response(content=buffer.getvalue(), media_type="audio/wav", headers=headers)
