# TTS (SpeechT5)

End-to-end Kinyarwanda TTS fine-tuning and inference project using SpeechT5.

## What This Project Does

- Loads and analyzes dataset speaker distribution
- Preprocesses audio/text for a selected speaker
- Fine-tunes `microsoft/speecht5_tts` with stage-wise training
- Exports deployable FP32 and INT8 inference packages
- Benchmarks latency and compares FP32 vs INT8

## Main Files

- `main.ipynb`: full Colab workflow (analysis -> preprocess -> train -> eval -> export)
- `src/training/speecht5_pipeline.py`: training data pipeline + trainer build + stage-wise run support
- `src/training/speecht5_inference.py`: synthesis, latency utilities, model export
- `infer.py`: local inference CLI and FP32/INT8 latency comparison
- `config.yaml`: active project configuration

## Colab Export Artifacts

From the notebook export cell, download:

- `speecht5_fp32_infer.zip`
- `speecht5_int8_deployment.zip`
- `speaker_embedding.pt`

These are the expected local artifacts for `infer.py`.

## Local Inference

Extract the model zip files (or pass zip paths directly), then run:

```bash
python infer.py --text "Muraho, nagufasha gute uyu munsi?" --mode both --out demo
```

Output WAVs are written to `wav_outputs/`:

- `wav_outputs/demo_fp32.wav`
- `wav_outputs/demo_int8.wav`

## Docker

Start the API with one command:

```bash
docker compose up --build
```

Alternative helper script:

```bash
bash docker/run.sh
```

After startup:

- API base: `http://localhost:8000`
- Health check: `GET /health`
- Synthesis: `POST /synthesize`

Example synthesis request:

```bash
curl -X POST "http://localhost:8000/synthesize" -H "Content-Type: application/json" -d "{\"text\":\"Muraho, nagufasha gute uyu munsi?\",\"mode\":\"fp32\"}" --output demo.wav
```

Docker expects local artifacts mounted by `docker-compose.yml`:

- `./speecht5_fp32_infer`
- `./speecht5_int8_deployment`
- `./speaker_embedding.pt`

## Notes

- INT8 in this project uses dynamic quantization on `nn.Linear` layers.
- Audio files are exported as PCM-16 WAV for broad player compatibility.
- CPU latency sometimes are above strict real-time targets; on GPU `<800ms> is met at short sentences.


## running swagger UI (local machine)

- set the paths.
        $env:TTS_FP32_DIR = (Resolve-Path .\speecht5_fp32_infer).Path
        $env:TTS_INT8_DIR = (Resolve-Path .\speecht5_int8_deployment).Path
        $env:TTS_SPEAKER_PATH = (Resolve-Path .\speaker_embedding.pt).Path
- run the command to launc the uvicorn server.
        uv run --active python -m uvicorn src.server:app --host 127.0.0.1 --port 8000

- you can also use the powershell command.
        @'
        {"text":"amakuru yawe?","mode":"fp32"}
        '@ | Set-Content -Path body.json -NoNewline

        curl.exe -sS -X POST "http://127.0.0.1:8000/synthesize" `
        -H "Content-Type: application/json" `
        --data-binary "@body.json" `
        --output demo_fp32.wav


        Format-Hex -Path .\demo_fp32.wav | Select-Object -First 1
        Get-Item .\demo_fp32.wav | Select-Object Name,Length