import os
import torch
import torchaudio
from datasets import load_dataset
from tqdm import tqdm
from src.config import (
    HF_DATASET_ID, 
    TARGET_SAMPLE_RATE, 
    PROCESSED_DIR, 
    MIN_DURATION, 
    MAX_DURATION
)

TARGET_SPEAKER_ID = 1 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Processing using device: {device}")

def preprocess_audio(waveform, sr):
    waveform = waveform.to(device)
    
    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE).to(device)
        waveform = resampler(waveform)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    max_val = torch.abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val * 0.9
        
    return waveform.cpu() 

def main():
    print(f"Loading {HF_DATASET_ID}...")
    dataset = load_dataset(HF_DATASET_ID, split="train", token=True)
    
    wavs_dir = os.path.join(PROCESSED_DIR, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    
    metadata = []
    dropped_counts = {"duration": 0, "speaker": 0, "empty_text": 0}
    
    print(f"Processing data for Speaker {TARGET_SPEAKER_ID} on {device}...")
    
    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        # 1. Filter by Speaker
        spk_id = item.get('speaker_id')
        if str(spk_id) != str(TARGET_SPEAKER_ID):
            dropped_counts["speaker"] += 1
            continue
            
        # 2. Extract Text
        text = item.get('text')
        if not text or len(text.strip()) == 0:
            dropped_counts["empty_text"] += 1
            continue

        # 3. Extract Audio
        audio_array = torch.tensor(item['audio']['array']).float().unsqueeze(0)
        orig_sr = item['audio']['sampling_rate']
        
        # 4. Duration Check
        duration = audio_array.shape[1] / orig_sr
        if duration < MIN_DURATION or duration > MAX_DURATION:
            dropped_counts["duration"] += 1
            continue
            
        # 5. Process on GPU
        processed_wav = preprocess_audio(audio_array, orig_sr)
        wav_int16 = (processed_wav * 32767).clamp(-32768, 32767).to(torch.int16)
        
        filename = f"kin_spk{TARGET_SPEAKER_ID}_{i:05d}.wav"
        save_path = os.path.join(wavs_dir, filename)
        torchaudio.save(save_path, wav_int16, TARGET_SAMPLE_RATE)
        
        # 7. Metadata
        metadata.append(f"{filename}|{text}")
        
    # Save Metadata CSV
    meta_path = os.path.join(PROCESSED_DIR, "metadata.csv")
    with open(meta_path, "w", encoding="utf-8") as f:
        for line in metadata:
            f.write(line + "\n")
            
    print("\nProcessing Complete!")
    print(f"   - Selected Speaker: {TARGET_SPEAKER_ID}")
    print(f"   - Saved Samples: {len(metadata)}")
    print(f"   - Dropped (Wrong Speaker): {dropped_counts['speaker']}")
    print(f"   - Dropped (Duration/Empty): {dropped_counts['duration'] + dropped_counts['empty_text']}")
    print(f"   - Data Location: {PROCESSED_DIR}")

if __name__ == "__main__":
    main()