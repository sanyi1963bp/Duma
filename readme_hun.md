# Voicetex Gemini Factory Ultra - Dark Mode Edition

Egy egyedi, helyben futó magyar nyelvű beszédfelismerő és diktáló szoftver, amely képes tanulni a felhasználó javításaiból (LoRA finomhangolással), valamint támogatja a kötegelt (batch) automata tanítást is hosszú hanganyagokból.

## Főbb funkciók
- **Push-to-Talk (PTT)** mód a `CTRL + WIN` kombinációval.
- **Sötét módú (Dark Mode)** modernizált Tkinter GUI.
- **HUD / Overlay** ablak az élő kivezérlésjelzőhöz (VU meter).
- **Helyi LoRA finomhangolás**: a javított szövegekből azonnal tanul, és automatikus `.zip` biztonsági mentést készít az adapterről.
- **Automata Gyár (Batch vágó)**: Hosszú `.wav` fájlok és kéziratok automatikus összeillesztése és tanítása.

## Követelmények
```bash
pip install torch transformers datasets peft numpy scipy sounddevice pyttsx3 librosa soundfile num2words keyboard