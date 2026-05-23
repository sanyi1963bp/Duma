\# Voicetex Ultra - Push-To-Talk AI Dictation \& Fine-Tuning Software



\*\*Voicetex Ultra\*\* is a professional, locally executable, Push-To-Talk (PTT) based Hungarian speech recognition and intelligent dictation application. This project combines OpenAI's state-of-the-art open-source speech recognition model with Low-Rank Adaptation (LoRA) fine-tuning technology, allowing the software to continuously adapt to the user's unique speech patterns, specialized vocabulary, and pronunciation.



I sincerely hope that this program brings joy to many users and significantly streamlines your daily workflow or digital content creation!



\---



\## 🚀 Key Features and Workflow

\* \*\*Push-To-Talk (PTT) Control:\*\* Recording is active only while holding down the `Ctrl + Win` hotkey combination. Releasing the keys instantly closes the audio stream and triggers the AI inference.

\* \*\*Automatic Clipboard \& Ingestion:\*\* The recognized text is copied to the Windows clipboard and automatically injected (pasted) directly into the active cursor position (e.g., Word, browser, text editors).

\* \*\*Live HUD \& VU-Meter:\*\* A borderless, floating overlay window stays on top of all applications, providing real-time visual feedback regarding the microphone levels and system state (Glowing Red: Recording, Yellow: AI processing, Dark Gray: Standby).

\* \*\*Interactive Learning Mode:\*\* If the AI mishears a word, you can easily correct it in the UI text box. Clicking the train button forces the software to immediately learn from your correction in the background.

\* \*\*Batch Training Factory:\*\* Allows slicing a long, clean audio file (.wav) alongside its exact transcript (.txt) for automated, bulk fine-tuning.



\---



\## 🤖 AI Models and Technical Specifications



\### Which LLM/Transformer model is used?

The software is powered by \*\*OpenAI's Whisper-Large-v3\*\* (`openai/whisper-large-v3`).

\* This does \*\*not\*\* rely on cloud-based APIs; the model runs \*\*100% locally on your machine\*\* (ideally accelerated via CUDA on an Nvidia GPU, though CPU execution is supported).

\* Whisper-Large-v3 is a 1.5-billion-parameter encoder-decoder Transformer model trained on millions of hours of multilingual audio, offering world-class performance for the Hungarian language.



\### Where are the models and checkpoints located?

\* \*\*Base Model:\*\* Upon first launch, the Hugging Face Transformers library automatically downloads the official weights from OpenAI's servers to your user cache directory (`\~/.cache/huggingface/hub/`).

\* \*\*Custom LoRA Adapters:\*\* The unique weights generated during fine-tuning are saved within the project root directory under the `./whisper\_lora\_magyar` folder. At startup, the application checks this directory and dynamically overlays the adapter onto the base model.



\### Mathematical \& AI Techniques Employed

1\. \*\*LoRA (Low-Rank Adaptation):\*\* Instead of retraining all 1.5 billion parameters, we freeze the base model and inject trainable, low-rank matrices into the Transformer's attention layers (`q\_proj`, `v\_proj`). This enables rapid and stable adaptation using short audio snippets.

2\. \*\*Audio Slicing (Librosa split):\*\* For batch processing, the software utilizes Librosa's RMS (Root-Mean-Square) energy-based thresholding to automatically detect silences and slice long audio files near sentence boundaries.

3\. \*\*Text Normalization \& Num2Words:\*\* Prior to matching audio data with transcripts, text is stripped of punctuation, converted to lowercase, and digits (e.g., "2026") are expanded into words ("húszezer-huszonhat") to ensure accurate synchronization checks.

4\. \*\*Difflib Sequence Matching:\*\* The script calculates a mathematical similarity score between the AI's transcription hypothesis and the actual text. If the ratio falls below 40%, the sample is rejected to prevent corrupt data from degrading the model.



\---



\## 💻 How to Use



\### Prerequisites and Installation

A Python 3.10+ environment is required. Install the necessary dependencies via command line:

```bash

pip install torch torchvision torchaudio --index-url \[https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

pip install transformers peft datasets sounddevice scipy librosa soundfile keyboard pyttsx3 num2words

Running the Application

Launch the script from your terminal:



Bash

python 2026-05-23\_06-30\_ptt\_stable\_hud\_voicetex.py

Daily Usage (Dictation)

Select your active microphone from the dropdown menu.



Click inside any target application (e.g., Word, Notepad, Browser chat) where you intend to type.



Press and HOLD DOWN the Ctrl + Win keys. The HUD overlay in the bottom right corner will turn red.



Speak continuously.



RELEASE the keys. The HUD changes to yellow, the AI decodes your voice, and the text is automatically pasted at your cursor!



⚖️ Legal Disclaimer \& Compliance

This project adheres to open-source community principles, utilizing exclusively legal and freely accessible technologies. Compliance with software licensing terms has been prioritized throughout development:



OpenAI Whisper-Large-v3: MIT License (Permissive use, modification, and distribution).



Hugging Face PEFT \& Transformers: Apache 2.0 License (Allows free private and commercial usage).



Python Modules: Integrated under MIT, BSD, and Apache 2.0 licensing models.



To the best of my knowledge and thorough review, this source code does not infringe upon any copyrights or intellectual property.



📞 Take-Down Policy

If you are a copyright holder or developer and find any license conflicts or compliance concerns within this repository, please DO NOT initiate immediate formal legal action. Open an "Issue" here on GitHub, and I guarantee that I will investigate the matter promptly and resolve, fix, or remove the flagged component as quickly as possible.



💬 Reviews, Ideas, and Development – Feedback Welcome!

I would be absolutely thrilled to hear your thoughts if you use this software! I am explicitly looking for feedback on the following:



Bug Reports: If you encounter any unexpected anomalies, crashes, or glitches.



Ideas and Feature Requests: What additional features would you like to see? (e.g., custom punctuation voice commands, alternative hotkeys, etc.)



Development Directions: How do you think the code could be further optimized, accelerated, or made more efficient?



Please share your thoughts under the GitHub Issues or Discussions tab! Let's build and improve this project together!

