import os
import torch
import torchaudio


TEST_SENTENCES = [
    "Muraho, nagufasha gute uyu munsi?",
    "Niba ufite ibibazo bijyanye n’ubuzima bwawe, twagufasha.",
    "Ni ngombwa ko ubonana umuganga vuba.",
]


@torch.no_grad()
def run_eval(
    model,
    tokenizer,
    device,
    output_dir,
    step
):

    os.makedirs(output_dir, exist_ok=True)

    model.eval()

    for i, text in enumerate(TEST_SENTENCES):

        inputs = tokenizer(
            text,
            return_tensors="pt"
        ).to(device)

        output = model(**inputs)

        waveform = output.waveform.cpu()

        path = os.path.join(
            output_dir,
            f"step_{step}_sample_{i}.wav"
        )

        torchaudio.save(
            path,
            waveform,
            model.config.sampling_rate
        )
