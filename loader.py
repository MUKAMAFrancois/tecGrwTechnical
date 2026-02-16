import sys
import os
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio
from src.config import HF_DATASET_ID, RAW_DIR, TARGET_SAMPLE_RATE
import getpass

def inspect_dataset():
    print(f"Authenticating and downloading {HF_DATASET_ID}...")

    # Prompt for Hugging Face token securely
    token = getpass.getpass("Enter your HuggingFace token: ")

    try:
        # Pass the token string directly
        dataset = load_dataset(HF_DATASET_ID, split="train", token=token)
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Dataset Loaded: {len(dataset)} samples")

    print("\nAnalyzing Metadata...")
    df = dataset.to_pandas()

    # Check columns
    print(f"Columns found: {df.columns.tolist()}")

    if 'speaker_id' in df.columns:
        speaker_counts = df['speaker_id'].value_counts()
        print("\nSpeaker Distribution:")
        print(speaker_counts)

        print("\n(Approximate Sample Counts per Speaker - Use this to decide strategy)")
    else:
        print("Warning: 'speaker_id' column missing! Check dataset format.")

    print("\n🎧 Inspecting first sample...")
    sample = dataset[0]
    audio_array = sample['audio']['array']
    sr = sample['audio']['sampling_rate']
    text = sample['text']

    print(f"Text: {text}")
    print(f"Sample Rate: {sr} Hz (Target: {TARGET_SAMPLE_RATE} Hz)")
    print(f"Shape: {audio_array.shape}")

    debug_path = os.path.join(RAW_DIR, "debug_sample_0.wav")
    sf.write(debug_path, audio_array, sr)
    print(f"Saved debug audio to: {debug_path}")
    print("Open this file in VS Code to listen for noise/quality.")

if __name__ == "__main__":
    inspect_dataset()