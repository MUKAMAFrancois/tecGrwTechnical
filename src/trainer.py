import os
import torch
import random
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from src.config import PROCESSED_DIR, TARGET_SAMPLE_RATE

# --- 1. Custom Formatter (FIXED) ---
def kin_formatter(root_path, manifest_file, **kwargs):
    """
    Reads the metadata file created by preprocessing.py.
    Format: filename.wav|text
    """
    items = []
    with open(manifest_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            cols = line.split("|")
            if len(cols) < 2:
                continue
                
            wav_filename = cols[0]
            text = cols[1]
            wav_path = os.path.join(root_path, "wavs", wav_filename)
            
            items.append({
                "text": text,
                "audio_file": wav_path,
                # --- FIX: ADD AUDIO UNIQUE NAME ---
                "audio_unique_name": wav_filename, 
                "speaker_name": "kin_spk1",
                "root_path": root_path,
                "language": "kin"
            })
    return items

def run_training(epochs=30):
    output_path = "/content/output"
    dataset_path = PROCESSED_DIR
    meta_file = os.path.join(dataset_path, "metadata.csv")
    
    # Dataset Config
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", 
        meta_file_train=meta_file,
        path=dataset_path 
    )
    
    # Audio Config
    audio_config = VitsAudioConfig(
        sample_rate=TARGET_SAMPLE_RATE,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None
    )

    # Model Config
    config = VitsConfig(
        audio=audio_config,
        run_name="kin_vits_production",
        batch_size=16,
        eval_batch_size=8,
        batch_group_size=4,
        
        # --- FIX: Reduce workers to prevent Colab freeze ---
        num_loader_workers=2,
        num_eval_loader_workers=2,
        
        # Training Steps
        run_eval=True,
        test_delay_epochs=0, # Start testing immediately (Epoch 0)
        epochs=epochs,
        lr_gen=2e-4, 
        lr_disc=2e-4,
        
        # Kinyarwanda Settings
        text_cleaner="basic_cleaners",
        use_phonemes=False, 
        compute_input_seq_cache=True,
        
        # Logging & Saving
        print_step=25,        # Print metrics every 25 steps
        print_eval=True,      # Print eval metrics
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        save_step=500,        # Save checkpoint frequently
        
        # Test Sentences (You will hear these in TensorBoard)
        test_sentences=[
            "Muraho, nagufasha gute uyu munsi?",
            "Niba ufite ibibazo bijyanye n'ubuzima bwawe, twagufasha.",
            "Ni ngombwa ko ubonana umuganga vuba."
        ]
    )

    tokenizer, config = TTSTokenizer.init_from_config(config)

    # Manual Data Loading
    print("📊 Loading Data manually with kin_formatter...")
    all_samples = kin_formatter(dataset_path, meta_file)
    
    if len(all_samples) == 0:
        raise ValueError("❌ No samples found! Check your metadata.csv and paths.")
    
    random.seed(42)
    random.shuffle(all_samples)
    
    # Split
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    eval_samples = all_samples[split_idx:]
    
    print(f"✅ Data Loaded: {len(train_samples)} Train, {len(eval_samples)} Eval")

    model = Vits(config, ap=AudioProcessor.init_from_config(config), tokenizer=tokenizer, speaker_manager=None)

    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print(f"🚀 Starting VITS Training for {epochs} epochs...")
    trainer.fit()

if __name__ == "__main__":
    run_training()