readme_content = """# Voicetex Gemini Factory Ultra - Dark Mode Edition

A custom, locally running Hungarian speech recognition and dictation software capable of learning from user corrections (via LoRA fine-tuning), featuring an automated batch factory for training from long audio materials.

## Key Features
- **Push-to-Talk (PTT) Mode**: Instant recording and typing by holding down the `CTRL + WIN` hotkey combination.
- **Dark Mode GUI**: Modernized, eye-strain-free dark user interface designed for comfortable night-time workflow.
- **HUD / Overlay Window**: A draggable, semi-transparent standalone widget with an active VU volume meter and status indicators that always stays on top.
- **Local LoRA Fine-Tuning**: The program instantly learns from your text corrections, updates the custom adapter, and generates automatic zip backups into the `voicetex_backups` directory.
- **Automated Factory (Batch Cutter)**: Automated alignment, intelligent silence filtering, Whisper-based synchronization checking, and batch training using long `.wav` audio files and precise text manuscripts.

## System Requirements
To run this software, a Python 3.10+ environment is required along with the following packages:
```bash
pip install torch transformers datasets peft numpy scipy sounddevice pyttsx3 librosa soundfile num2words keyboard