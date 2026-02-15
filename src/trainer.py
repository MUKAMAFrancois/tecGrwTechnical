import os
import torch
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from src.config import PROCESSED_DIR, TARGET_SAMPLE_RATE

def run_training(epochs=30):
    output_path = "/content/output"
    dataset_path = PROCESSED_DIR
    meta_file = os.path.join(dataset_path, "metadata.csv")
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", 
        meta_file_train=meta_file,
        path=os.path.join(dataset_path, "wavs")
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
        batch_size=32,   
        eval_batch_size=8,
        batch_group_size=4,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        
        # Training Steps
        run_eval=True,
        test_delay_epochs=-1, 
        epochs=epochs,
        lr_gen=2e-4, 
        lr_disc=2e-4,
        
        text_cleaner="basic_cleaners",
        use_phonemes=True,
        phoneme_language="run",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        compute_input_seq_cache=True,
        
        # Logging & Saving
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        save_step=1000, 
        
        test_sentences=[
            "Muraho, nagufasha gute uyu munsi?",
            "Niba ufite ibibazo bijyanye n'ubuzima bwawe, twagufasha.",
            "Ni ngombwa ko ubonana umuganga vuba.",
            "Twabanye nawe kandi tuzakomeza kukwitaho.",
            "Ushobora kuduhamagara igihe cyose ukeneye ubufasha."
        ]
    )

    tokenizer, config = TTSTokenizer.init_from_config(config)

    # Load Data
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=0.1,
        eval_split_size=0.1,
    )

    model = Vits(config, ap=AudioProcessor.init_from_config(config), tokenizer=tokenizer, speaker_manager=None)

    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print(f"Starting VITS Training for {epochs} epochs...")
    trainer.fit()

if __name__ == "__main__":
    run_training()