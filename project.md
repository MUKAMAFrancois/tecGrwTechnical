# Tekana Project
## Data Scientist Take-Home Assessment
### Kinyarwanda Text-to-Speech Challenge

---

## Overview

- **Duration:** Due Tonight.
- **Dataset:** 14 hours of Kinyarwanda audio *(link will be provided)*
- Generative AI tools are allowed, but you must clearly state how you used them.
- **Submission:** GitHub repository and a short PDF write-up.

---

## Dataset Description

> **Confidentiality notice:** This dataset is confidential and the sole property of Tecgrw. It must not be shared, redistributed, published, or used outside the scope of this assessment.

The dataset provided for this challenge consists of approximately 14 hours of high-quality, preprocessed Kinyarwanda speech recordings. The audio has already undergone initial quality pre-processing. More than 12 million people speak Kinyarwanda, and commercial text-to-speech systems poorly support it. Candidates are free to apply any additional pre-processing steps they find appropriate and to document them in their write-up.

### Content and Coverage

The recordings cover a broad range of topics and domains, spanning fields such as science, health, education, politics, and everyday conversation. This topical diversity is intentional: it ensures that the resulting TTS voice is robust across the kind of varied language a real caller to the Tekana health line might encounter.

### Speakers

The dataset features **three distinct speakers**, identified by the `speaker_id` column in the metadata file. This multi-speaker composition gives candidates the opportunity to either train a single averaged voice or explore speaker-conditioned approaches. You should clearly state which strategy you chose and why.

### Metadata Format

Each audio clip is described by a metadata CSV file with the following four columns:

| Column | Description |
|---|---|
| `audio` | Recording file path |
| `txt` | Ground-truth Kinyarwanda transcription of the audio clip |
| `speaker_id` | Speaker identifier, indicating which of the speakers produced the recording |

### Data Access

The dataset is hosted on Hugging Face at:

```
Washere-1/tecgrw-audio
```

Access credentials or a dataset link will be communicated separately if the repository is private.

---

## Permitted Use and Expectations

You are free to process, filter, segment, augment, or otherwise transform the data as you judge appropriate for the task. There is no single prescribed pipeline. However, you must:

- Document every processing step you apply in your write-up.
- Report any data quality issues you discovered (noise, misaligned transcripts, clipping, silence, etc.) and explain how you handled them.
- Not share, upload to external services, or use the data for any purpose other than this assessment.

---

## 1. Background

Tekana is a voice-based AI system that answers phone calls from young people in Rwanda who are looking for guidance on sexual and mental health. A caller speaks naturally, sometimes in Kinyarwanda. The system listens, understands, and prepares a helpful response.

Two parts of the system already work:
1. **Speech-to-text** converts the caller's voice into written text.
2. **The AI** generates a safe and appropriate response.

What is missing is the final step: turning that response back into speech. Right now, when Tekana is ready to answer, there is silence. **Your task is to build the voice.**

---

## 2. The Challenge

Kinyarwanda is spoken by more than 12 million people, but it is poorly supported by commercial text-to-speech systems. Many existing systems sound unnatural or clearly foreign. For a health support line, voice quality matters. If the voice sounds robotic or mispronounces words, people may lose trust.

There are also production constraints:

- Inference latency must be **under 800 ms** for a short (~10-word) sentence.
- The model should ideally be **under 200 MB** on disk (hard limit: 1 GB).
- The voice should be **clearly understandable** to a Rwandan listener. A perfect accent is not required, but it should not feel foreign or robotic.
- **No user voice data** may be stored or sent to external APIs.
- You only have **14 hours** of Kinyarwanda audio. It is suitable for direct use in TTS training pipelines. We are not expecting perfection — we are looking for strong reasoning and practical decision-making.

---

## 3. What You Need to Do

### 3.1 Choose a Base Model

We strongly recommend starting from **`facebook/mms-tts`**, which already includes a Kinyarwanda checkpoint and has a relatively small footprint. You may choose another approach (e.g., Piper TTS, Coqui TTS, or training VITS from scratch), but you must clearly justify your decision. Your reasoning should be especially convincing. Explain:

- What you chose
- What you seriously considered
- Why you made that decision given the size and latency constraints

### 3.2 Prepare the Data

Before training, you should:

- Resample audio to the model's required sample rate
- Segment audio into 2–12 second clips
- Normalize volume
- Generate or clean transcriptions (Whisper is allowed)
- Create a metadata file mapping each audio file to its transcript
- *(Note: The dataset is already split)*

If you find issues in the data (noise, incorrect transcripts, clipping, etc.), describe them and explain how you handled them.

### 3.3 Fine-Tune or Train

- Use reproducible code.
- Store hyperparameters in a configuration file.
- During training, periodically synthesize the same fixed sentences so you can listen to progress.
- Stop training when the audio quality is good enough — not just when the loss decreases.

### 3.4 Evaluate

Measure your own **subjective listening evaluation**.

You must synthesize and include audio for the following sentences:

- *"Muraho, nagufasha gute uyu munsi?"*
- *"Niba ufite ibibazo bijyanye n'ubuzima bwawe, twagufasha."*
- *"Ni ngombwa ko ubonana umuganga vuba."*
- *"Twabanye nawe kandi tuzakomeza kukwitaho."*
- *"Ushobora kuduhamagara igihe cyose ukeneye ubufasha."*

### 3.5 Package for Production

Provide:

- A simple **inference script** that takes a text string and outputs a WAV file
- Printed **latency** for each call
- A **`requirements.txt`** file
- A **Docker setup** that runs with a single command or an API

---

## 4. The Write-Up

Your written explanation matters as much as your code. Clearly describe:

- Why you selected your model
- Why you fine-tuned instead of training from scratch (or vice versa)
- How you prepared the data
- Your main training decisions (learning rate, batch size, stopping criteria)
- Why your evaluation approach makes sense for this use case

Also **list every generative AI tool you used** and how you used it.

---

## 5. How We Will Evaluate You

We will evaluate your submission based on:

- **Audio quality** — Is the speech understandable and natural enough for real users?
- **Latency and efficiency** — Does it meet performance and size constraints?

You will present your work in a **30-minute technical review**. Be prepared to run inference live and explain your decisions.

The dataset is available at **`Washere-1/tecgrw-audio`** on Hugging Face.

