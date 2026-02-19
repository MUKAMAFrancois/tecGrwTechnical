import inspect
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import Audio, Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
)


def resolve_metadata_path(cfg, key):
    raw = cfg.get(key)
    if raw is None:
        return None
    if os.path.exists(raw):
        return raw
    return os.path.join(cfg.get("PROCESSED_DIR", ""), os.path.basename(raw))


def load_tts_dataset(metadata_path, target_sr=16000):
    if metadata_path is None:
        raise ValueError("metadata_path is None.")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(
        metadata_path,
        sep="|",
        header=None,
        names=["audio", "text", "speaker_id"],
    )
    df = df[df["text"].notna()].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df[df["audio"].map(os.path.exists)]

    ds = Dataset.from_pandas(df[["audio", "text"]], preserve_index=False)
    ds = ds.cast_column("audio", Audio(sampling_rate=target_sr))
    return ds


def load_train_val_datasets(config):
    train_meta = resolve_metadata_path(config, "TRAIN_METADATA")
    val_meta = resolve_metadata_path(config, "VAL_METADATA")

    train_ds = load_tts_dataset(train_meta, int(config["TARGET_SAMPLE_RATE"]))
    val_ds = load_tts_dataset(val_meta, int(config["TARGET_SAMPLE_RATE"]))

    if len(train_ds) == 0:
        raise ValueError(f"Train dataset is empty after filtering: {train_meta}")
    if len(val_ds) == 0:
        raise ValueError(
            f"Validation dataset is empty after filtering: {val_meta}. "
            "No evaluation metrics (including eval_loss) can be produced."
        )

    return train_ds, val_ds


def load_speecht5_components(device):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    return processor, model, vocoder


def get_speaker_embedding(train_dataset, device):
    try:
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda backend: None

        from speechbrain.inference.speaker import EncoderClassifier

        spk_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            run_opts={"device": str(device)},
            savedir="pretrained_spkrec",
        )

        wav = train_dataset[0]["audio"]["array"]
        wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = spk_encoder.encode_batch(wav).squeeze().detach().cpu().numpy().astype(np.float32)

        print("Using SpeechBrain speaker embedding:", emb.shape)
        return emb
    except Exception as exc:
        print("Speaker encoder unavailable. Error:", exc)
        print("Falling back to 'Matthijs/cmu-arctic-xvectors' (index 7306).")
        from datasets import load_dataset
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        fallback_emb = torch.tensor(embeddings_dataset[7306]["xvector"]).numpy().astype(np.float32)
        return fallback_emb


def normalize_input_ids(input_ids):
    arr = np.asarray(input_ids)
    if arr.ndim == 0:
        raise ValueError("input_ids collapsed to scalar")
    if arr.ndim == 2:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"Unexpected input_ids shape: {arr.shape}")
    return arr.astype(np.int64).tolist()


def normalize_labels(labels, n_mels):
    arr = np.asarray(labels)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected labels shape: {arr.shape}")

    if arr.shape[-1] == n_mels:
        pass
    elif arr.shape[0] == n_mels:
        arr = arr.T
    else:
        raise ValueError(f"Could not align labels to n_mels={n_mels}, got {arr.shape}")

    return arr.astype(np.float32)


def build_prepare_example_fn(processor, model, speaker_embedding):
    n_mels = int(model.config.num_mel_bins)

    def prepare_example(batch):
        audio = batch["audio"]
        out = processor(
            text=batch["text"],
            audio_target=audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=False,
        )

        input_ids = normalize_input_ids(out["input_ids"])
        labels = normalize_labels(out["labels"], n_mels)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "speaker_embeddings": speaker_embedding,
        }

    return prepare_example


def build_processed_datasets(train_ds, val_ds, processor, model, speaker_embedding):
    prepare_example = build_prepare_example_fn(processor, model, speaker_embedding)
    train_proc = train_ds.map(prepare_example, remove_columns=train_ds.column_names)
    val_proc = val_ds.map(prepare_example, remove_columns=val_ds.column_names)
    return train_proc, val_proc


class TTSDataCollatorWithPadding:
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model
        self.n_mels = int(model.config.num_mel_bins)

    def __call__(self, features):
        input_ids = [{"input_ids": f["input_ids"]} for f in features]
        label_features = [{"input_values": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_ids=input_ids,
            labels=label_features,
            return_tensors="pt",
        )

        batch["input_ids"] = batch["input_ids"].long()

        labels = batch["labels"]
        if labels.dim() != 3:
            raise RuntimeError(f"Expected 3D labels, got {labels.shape}")

        if labels.shape[-1] == self.n_mels:
            pass
        elif labels.shape[1] == self.n_mels:
            labels = labels.transpose(1, 2)
        else:
            raise RuntimeError(f"Unexpected labels shape: {labels.shape}")

        dec_mask = batch.pop("decoder_attention_mask", None)
        if dec_mask is not None:
            t = min(labels.shape[1], dec_mask.shape[1])
            labels = labels[:, :t, :]
            dec_mask = dec_mask[:, :t]
            labels = labels.masked_fill(dec_mask.unsqueeze(-1).ne(1), -100)

        rf = int(self.model.config.reduction_factor)
        if rf > 1:
            if dec_mask is not None:
                lengths = (dec_mask.sum(dim=1) // rf) * rf
                max_len = max(1, int(lengths.max().item()))
            else:
                max_len = (labels.shape[1] // rf) * rf
                max_len = max(1, int(max_len))
            labels = labels[:, :max_len, :]

        batch["labels"] = labels
        batch["speaker_embeddings"] = torch.tensor(
            [f["speaker_embeddings"] for f in features],
            dtype=torch.float32,
        )
        return batch


def print_preprocessed_batch_debug(train_proc, data_collator):
    print("First processed sample checks:")
    print("  input_ids len:", len(train_proc[0]["input_ids"]))
    print("  labels shape:", np.asarray(train_proc[0]["labels"]).shape)

    sample_n = min(2, len(train_proc))
    sample_batch = data_collator([train_proc[i] for i in range(sample_n)])
    print("Batch sanity check:")
    for key, value in sample_batch.items():
        print(f"  {key}: {tuple(value.shape)}")


@dataclass
class TrainerBundle:
    args: Seq2SeqTrainingArguments
    trainer: Seq2SeqTrainer
    output_dir: str
    stages: List[Dict[str, Any]]


def parse_training_stages(config):
    raw_stages = (config or {}).get("STAGES") or []
    stages = []

    for idx, raw in enumerate(raw_stages, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"STAGES[{idx - 1}] must be a dict, got: {type(raw)}")

        stage_id = int(raw.get("stage", idx))
        lr = float(raw["lr"])
        epochs = float(raw["epochs"])

        if lr <= 0:
            raise ValueError(f"STAGES[{idx - 1}] has invalid lr={lr}. Must be > 0.")
        if epochs <= 0:
            raise ValueError(f"STAGES[{idx - 1}] has invalid epochs={epochs}. Must be > 0.")

        stages.append({"stage": stage_id, "lr": lr, "epochs": epochs})

    return stages


def _set_eval_strategy(kwargs, value):
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "eval_strategy" in sig:
        kwargs["eval_strategy"] = value
    elif "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = value
    else:
        raise RuntimeError("Could not find eval strategy parameter in Seq2SeqTrainingArguments.")


def _set_if_supported(kwargs, key, value):
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if key in sig:
        kwargs[key] = value


def build_trainer_bundle(
    model,
    processor,
    train_proc,
    val_proc,
    data_collator,
    config=None,
    output_dir="speecht5_finetuned",
):
    if len(train_proc) == 0:
        raise ValueError("train_proc is empty.")
    if len(val_proc) == 0:
        raise ValueError(
            "val_proc is empty. With load_best_model_at_end=True, no eval metrics can be produced."
        )

    stages = parse_training_stages(config)
    if stages:
        initial_lr = float(stages[0]["lr"])
        initial_epochs = float(stages[0]["epochs"])
    else:
        initial_lr = 2e-5
        initial_epochs = 10

    base_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=initial_lr,
        warmup_steps=500,
        num_train_epochs=initial_epochs,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        do_eval=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        save_strategy="epoch",
        label_names=["labels"],
    )

    _set_eval_strategy(base_kwargs, "epoch")
    _set_if_supported(base_kwargs, "predict_with_generate", False)
    _set_if_supported(base_kwargs, "eval_on_start", True)

    args = Seq2SeqTrainingArguments(**base_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_proc,
        eval_dataset=val_proc,
        data_collator=data_collator,
    )

    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor.tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)
    if not getattr(trainer, "label_names", None):
        trainer.label_names = ["labels"]
    return TrainerBundle(args=args, trainer=trainer, output_dir=output_dir, stages=stages)


def run_stagewise_training(bundle):
    trainer = bundle.trainer
    stages = bundle.stages

    if not stages:
        print(
            "No STAGES found in config. Running single training pass with "
            f"lr={trainer.args.learning_rate}, epochs={trainer.args.num_train_epochs}."
        )
        return trainer.train()

    stage_results = []
    for stage_cfg in stages:
        stage_id = int(stage_cfg["stage"])
        stage_lr = float(stage_cfg["lr"])
        stage_epochs = float(stage_cfg["epochs"])

        trainer.args.learning_rate = stage_lr
        trainer.args.num_train_epochs = stage_epochs

        # Recreate optimizer/scheduler so each stage starts with its own LR setup.
        trainer.optimizer = None
        trainer.lr_scheduler = None

        print(
            f"\n[Stage {stage_id}] training for {stage_epochs} epochs at lr={stage_lr:.8f}"
        )
        stage_results.append(trainer.train())

    return stage_results[-1] if stage_results else None
