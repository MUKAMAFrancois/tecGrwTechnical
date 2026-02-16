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
            
            # The wavs are in PROCESSED_DIR/wavs
            wav_path = os.path.join(root_path, "wavs", wav_filename)
            
            items.append({
                "text": text,
                "audio_file": wav_path,
                "speaker_name": "kin_spk1",
                "root_path": root_path,
                "language": "kin"
            })
    return items

def run_training(epochs=30):
    output_path = "/content/output"
    dataset_path = PROCESSED_DIR
    meta_file = os.path.join(dataset_path, "metadata.csv")
    

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", 
        meta_file_train=meta_file,
        path=dataset_path 
    )
    
    audio_config = VitsAudioConfig(
        sample_rate=TARGET_SAMPLE_RATE,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None
    )

    config = VitsConfig(
        audio=audio_config,
        run_name="kin_vits_production",
        batch_size=16,
        eval_batch_size=8,
        batch_group_size=4,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1, 
        epochs=epochs,
        lr_gen=2e-4, 
        lr_disc=2e-4,
        text_cleaner="basic_cleaners",
        use_phonemes=False, # Critical for Kinyarwanda
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        save_step=1000, 
        test_sentences=[
            "Muraho, nagufasha gute uyu munsi?",
            "Niba ufite ibibazo bijyanye n'ubuzima bwawe, twagufasha.",
            "Ni ngombwa ko ubonana umuganga vuba."
        ]
    )

    # 2. Tokenizer
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # 3. MANUAL DATA LOADING (The Fix)
    print("📊 Loading Data manually with kin_formatter...")
    all_samples = kin_formatter(dataset_path, meta_file)
    
    # Check if we found samples
    if len(all_samples) == 0:
        raise ValueError("❌ No samples found! Check your metadata.csv and paths.")
    
    # Shuffle and Split (90% Train, 10% Eval)
    random.seed(42)
    random.shuffle(all_samples)
    
    eval_split_size = 0.1
    split_idx = int(len(all_samples) * (1 - eval_split_size))
    
    train_samples = all_samples[:split_idx]
    eval_samples = all_samples[split_idx:]
    
    print(f"✅ Data Loaded: {len(train_samples)} Train, {len(eval_samples)} Eval")

    # 4. Initialize Model
    model = Vits(config, ap=AudioProcessor.init_from_config(config), tokenizer=tokenizer, speaker_manager=None)

    # 5. Initialize Trainer
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples, # Pass the lists directly
        eval_samples=eval_samples,
    )

    print(f"🚀 Starting VITS Training for {epochs} epochs...")
    trainer.fit()

if __name__ == "__main__":
    run_training()