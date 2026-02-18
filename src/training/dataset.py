import os
import math
import random
import torch
import torchaudio
import pandas as pd

from torch.utils.data import Dataset, DataLoader, Sampler


class TTSDataset(Dataset):
    """
    Dataset for loading processed TTS audio and text.
    """

    def __init__(self, metadata_path, tokenizer, target_sr):
        self.df = pd.read_csv(
            metadata_path,
            sep="|",
            header=None,
            names=["audio_path", "text", "speaker_id"]
        )

        self.tokenizer = tokenizer
        self.target_sr = target_sr
        self.durations = self._compute_durations()

    def _compute_durations(self):

        durations = []

        for audio_path in self.df["audio_path"].tolist():
            try:
                info = torchaudio.info(audio_path)
                duration = info.num_frames / float(info.sample_rate)
            except Exception:
                waveform, sr = torchaudio.load(audio_path)
                duration = waveform.shape[-1] / float(sr)

            durations.append(max(duration, 1e-4))

        return durations

    def get_duration(self, idx):
        return self.durations[idx]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        audio_path = row["audio_path"]
        text = row["text"]

        waveform, sr = torchaudio.load(audio_path)

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        input_ids = self.tokenizer(
            text,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        return {
            "waveform": waveform,
            "input_ids": input_ids,
        }


class DurationBatchSampler(Sampler):
    """
    Build variable-size batches capped by total audio duration.
    """

    def __init__(self, dataset, max_batch_duration_sec, shuffle=True):
        self.dataset = dataset
        self.max_batch_duration_sec = float(max_batch_duration_sec)
        self.shuffle = shuffle

        if self.max_batch_duration_sec <= 0:
            raise ValueError("max_batch_duration_sec must be > 0.")

    def _iter_indices(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        return indices

    def __iter__(self):
        batch = []
        batch_duration = 0.0

        for idx in self._iter_indices():
            sample_duration = float(self.dataset.get_duration(idx))

            if batch and (batch_duration + sample_duration) > self.max_batch_duration_sec:
                yield batch
                batch = []
                batch_duration = 0.0

            batch.append(idx)
            batch_duration += sample_duration

        if batch:
            yield batch

    def __len__(self):
        total_duration = sum(self.dataset.durations)
        return max(1, math.ceil(total_duration / self.max_batch_duration_sec))


class TTSDataCollator:
    """
    Pads batch correctly.
    """

    def __call__(self, batch):

        waveforms = [item["waveform"].squeeze(0) for item in batch]
        input_ids = [item["input_ids"] for item in batch]

        waveforms = torch.nn.utils.rnn.pad_sequence(
            waveforms,
            batch_first=True
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=0
        )

        return {
            "waveforms": waveforms.unsqueeze(1),
            "input_ids": input_ids,
        }


def create_dataloader(
    metadata_path,
    tokenizer,
    target_sr,
    batch_size,
    shuffle=True,
    num_workers=2,
    max_batch_duration_sec=None
):

    dataset = TTSDataset(
        metadata_path,
        tokenizer,
        target_sr
    )

    collator = TTSDataCollator()

    if max_batch_duration_sec is not None:
        batch_sampler = DurationBatchSampler(
            dataset,
            max_batch_duration_sec=max_batch_duration_sec,
            shuffle=shuffle
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=num_workers
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            num_workers=num_workers
        )

    return loader
