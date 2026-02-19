# src/loader.py

import os
import yaml
import getpass

from datasets import load_dataset, Audio,concatenate_datasets

def load_config(config_path=None):
    """
    Load YAML config file.
    """

    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "config.yaml"
        )

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_hf_token():
    """
    Get HuggingFace token.
    """
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
        if token:
            print("Using HF token from Colab secrets")
            return token
    except (ImportError, ModuleNotFoundError, Exception):
        pass

    token = os.environ.get("HF_TOKEN")
    if token:
        print("Using HF token from environment variable")
        return token

    return getpass.getpass("Enter HuggingFace token: ")


def load_dataset_split(config, split, token=None):
    """
    Load a single dataset split.

    Args:
        config (dict)
        split (str): "train", "validation", or "test"
        token (str, optional)

    Returns:
        HuggingFace Dataset
    """

    hf_dataset_id = config["HF_DATASET_ID"]
    target_sr = config["TARGET_SAMPLE_RATE"]

    if token is None:
        token = get_hf_token()

    dataset = load_dataset(
        hf_dataset_id,
        split=split,
        token=token
    )

    dataset = dataset.cast_column(
        "audio",
        Audio(sampling_rate=target_sr)
    )

    print(f"Loaded {split} split: {len(dataset)} samples")
    for i in range(min(3, len(dataset))):
        text = dataset[i].get("text", "")
        print(f"  [{i+1}] {text[:50]}")

    return dataset


def load_all_splits(config, token=None):
    """
    Load train, validation, and test splits.

    Returns:
        dict:
            {
                "train": Dataset,
                "validation": Dataset,
                "test": Dataset
            }
    """

    splits = {}

    for split in ["train", "validation", "test"]:
        splits[split] = load_dataset_split(
            config,
            split,
            token=token
        )

    return splits




def combine_splits(splits_dict):
    return concatenate_datasets(list(splits_dict.values()))


def ensure_dir(path):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def get_sample_duration(sample):
    """
    Compute duration of one sample in seconds.
    """

    audio = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]

    return len(audio) / sr

def filter_speaker(dataset_split, speaker_id):
    """
    Takes a HF dataset split, converts to pandas, and filters by speaker_id.
    """
    df = dataset_split.to_pandas()
    
    df["speaker_id"] = df["speaker_id"].astype(int)
    
    df_filtered = df[df["speaker_id"] == int(speaker_id)].reset_index(drop=True)
    return df_filtered


def load_raw_splits(config):
    """
    Loads the dataset and returns a dictionary of filtered pandas DataFrames 
    keys: 'train', 'validation', 'test'
    """
    dataset_name = config["HF_DATASET_ID"]
    target_speaker = config["SELECTED_SPEAKER_ID"]
    
    print(f"[Loader] Loading {dataset_name} from Hugging Face...")
    
    dataset_dict = load_dataset(dataset_name)
    
    processed_splits = {}
    
    for split_name in dataset_dict.keys():
        print(f"[Loader] Processing split: {split_name}")
        df_filtered = filter_speaker(dataset_dict[split_name], target_speaker)
        processed_splits[split_name] = df_filtered
        print(f"   -> Found {len(df_filtered)} samples for Speaker {target_speaker}")

    return processed_splits

def extract_sample_fields(sample):
    """
    Extract standardized fields from sample.

    Returns:
        dict:
            audio_array
            sampling_rate
            text
            speaker_id
            duration_sec
    """

    audio_array = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]

    text = sample.get("text", None)
    speaker_id = sample.get("speaker_id", None)

    duration_sec = len(audio_array) / sr

    return {
        "audio_array": audio_array,
        "sampling_rate": sr,
        "text": text,
        "speaker_id": speaker_id,
        "duration_sec": duration_sec
    }
