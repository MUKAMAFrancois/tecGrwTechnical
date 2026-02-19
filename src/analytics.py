# src/analytics.py

import pandas as pd
from tqdm import tqdm

from src.loader import (
    load_all_splits,
    combine_splits,
    get_sample_duration,
)

def compute_speaker_stats(dataset):
    """
    Compute speaker statistics.

    Args:
        dataset: iterable dataset

    Returns:
        DataFrame
    """

    records = []

    for sample in tqdm(dataset, desc="Analyzing speakers"):

        speaker = sample.get("speaker_id", "unknown")

        duration_sec = get_sample_duration(sample)

        records.append({
            "speaker_id": speaker,
            "duration_sec": duration_sec
        })

    df = pd.DataFrame(records)

    stats = df.groupby("speaker_id").agg(
        total_duration_sec=("duration_sec", "sum"),
        mean_duration_sec=("duration_sec", "mean"),
        median_duration_sec=("duration_sec", "median"),
        clip_count=("duration_sec", "count")
    ).reset_index()

    stats["total_duration_hr"] = stats["total_duration_sec"] / 3600

    stats = stats.sort_values(
        by="total_duration_sec",
        ascending=False
    )

    return stats


def recommend_speaker(stats_df):
    """
    Recommend best speaker.

    Criteria:
        highest total duration
    """

    best = stats_df.iloc[0]

    return best["speaker_id"]


def save_speaker_report(stats_df, path="evaluation/speaker_report.csv"):

    stats_df.to_csv(path, index=False)


def run_speaker_analysis(config, token=None):
    """
    Run analysis across ALL splits.

    Returns:
        stats_df
        best_speaker_id
    """

    splits = load_all_splits(config, token)

    combined_dataset = combine_splits(splits)

    stats = compute_speaker_stats(combined_dataset)

    best_speaker = recommend_speaker(stats)

    return stats, best_speaker


def analyze_each_split(config, token=None):
    """
    Returns stats per split.
    """

    splits = load_all_splits(config, token)

    results = {}

    for split_name, dataset in splits.items():

        stats = compute_speaker_stats(dataset)

        results[split_name] = stats

    return results
