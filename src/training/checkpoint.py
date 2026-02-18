import os


def save_checkpoint(
    model,
    tokenizer,
    output_dir,
    step
):

    path = os.path.join(output_dir, f"checkpoint_{step}")

    os.makedirs(path, exist_ok=True)

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    print(f"Saved checkpoint: {path}")
