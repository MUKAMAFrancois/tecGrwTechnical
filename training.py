import os

from src.loader import load_config
from src.model import load_model_and_tokenizer, get_device
from src.training.dataset import create_dataloader
from src.training.trainer import Trainer


def resolve_metadata_path(config, key):

    raw_path = config.get(key)
    if raw_path is None:
        return None

    if os.path.exists(raw_path):
        return raw_path

    processed_dir = config.get("PROCESSED_DIR", "")
    candidate = os.path.join(processed_dir, os.path.basename(raw_path))

    if os.path.exists(candidate):
        return candidate

    return raw_path


def main():

    config = load_config()

    device = get_device()

    print("Device:", device)

    model, tokenizer = load_model_and_tokenizer(
        config,
        device
    )

    train_metadata = resolve_metadata_path(config, "TRAIN_METADATA")
    val_metadata = resolve_metadata_path(config, "VAL_METADATA")

    train_loader = create_dataloader(
        train_metadata,
        tokenizer,
        config["TARGET_SAMPLE_RATE"],
        config["BATCH_SIZE"],
        shuffle=True,
        num_workers=int(config.get("NUM_WORKERS", 2)),
        max_batch_duration_sec=config.get("MAX_BATCH_DURATION_SEC")
    )

    val_loader = None
    if val_metadata is not None and os.path.exists(val_metadata):
        val_loader = create_dataloader(
            val_metadata,
            tokenizer,
            config["TARGET_SAMPLE_RATE"],
            config["BATCH_SIZE"],
            shuffle=False,
            num_workers=int(config.get("NUM_WORKERS", 2)),
            max_batch_duration_sec=config.get("VAL_MAX_BATCH_DURATION_SEC")
        )
        print("Validation loader enabled.")
    else:
        print("Validation metadata not found, validation disabled.")

    trainer = Trainer(
        model,
        tokenizer,
        train_loader,
        val_loader,
        config,
        device,
        use_amp=bool(config.get("USE_AMP", True))
    )

    stages = config.get("STAGES", [
        {"stage": 1, "lr": 1e-4, "epochs": 3},
        {"stage": 2, "lr": 5e-5, "epochs": 2},
        {"stage": 3, "lr": 1e-5, "epochs": 1},
    ])

    for stage_cfg in stages:
        trainer.train_stage(
            stage=int(stage_cfg["stage"]),
            lr=float(stage_cfg["lr"]),
            epochs=int(stage_cfg["epochs"])
        )


if __name__ == "__main__":

    main()
