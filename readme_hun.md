# Voicetex Gemini Factory Ultra - Dark Mode Edition

Egy egyedi, helyben futó magyar nyelvű beszédfelismerő és diktáló szoftver, amely képes tanulni a felhasználó javításaiból (LoRA finomhangolással), valamint támogatja a kötegelt (batch) automata tanítást is hosszú hanganyagokból.

## Főbb funkciók
- **Push-to-Talk (PTT) mód**: Gyors rögzítés és gépelés a `CTRL + WIN` billentyűkombó nyomva tartásával.
- **Sötét módú GUI**: Modernizált, szemkímélő sötét felület a kényelmes éjszakai munkához.
- **HUD / Overlay ablak**: Külön mozgatható, félig áttetsző kis ablak élő kivezérlésjelzővel (VU meter) és állapotjelzéssel, ami mindig legfelül marad.
- **Helyi LoRA finomhangolás**: A javított szövegekből a program azonnal tanul, frissíti az egyedi adaptert, és automatikus zip biztonsági mentést készít róla a `voicetex_backups` mappába.
- **Automata Gyár (Batch vágó)**: Hosszú `.wav` fájlok és pontos kéziratok automatikus összeillesztése, intelligens csend-szűrése, Whisper-alapú szinkronellenőrzése és kötegelt tanítása.

## Rendszerkövetelmények
A futtatáshoz Python 3.10+ környezet és az alábbi könyvtárak szükségesek:
```bash
pip install torch transformers datasets peft numpy scipy sounddevice pyttsx3 librosa soundfile num2words keyboard