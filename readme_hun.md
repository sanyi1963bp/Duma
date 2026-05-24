# Voicetex Gemini Factory Ultra - Dark Mode Edition

A Voicetex Gemini egy professzionális, teljes mértékben helyben futó (offline), magyar nyelvű beszédfelismerő és diktáló szoftver. A rendszer egyedisége, hogy képes tanulni a felhasználó saját javításaiból a beépített helyi LoRA (Low-Rank Adaptation) fine-tuning technológiának köszönhetően. Emellett tartalmaz egy automatizált „Gyár” (Batch) modult, amely hosszú hanganyagok és kéziratok alapján képes tömeges, felügyelt tanító-adatbázist előállítani.

Ez a projekt ideális választás azoknak, akiknek fontos az adatbiztonság (nem küld hangot felhős szerverekre), és egy idővel egyre pontosabbá váló, személyre szabott diktáló társat szeretnének.

---

## Főbb funkciók és modulok

### 🎙️ Intenzív Diktálás & Push-to-Talk (PTT)
- **Azonnali rögzítés**: A `CTRL + WIN` billentyűkombináció nyomva tartásával a szoftver azonnal rögzíti a hangot.
- **Automata gépelés és vágólap**: Az elengedés pillanatában a mesterséges intelligencia szöveggé alakítja a beszédet, beírja a kurzor aktuális pozíciójához (szimulált billentyűzet-bevitellel), és másolatot készít a vágólapra is.

### 🎚️ Élő HUD & Overlay Ablak
- **Mindig legfelül (Always on Top)**: Egy kisméretű, félig áttetsző, egérrel bárhová elhúzható lebegő ablak, amely munka közben sem zavarja a látványt.
- **Élő VU-meter**: Beépített kivezérlésjelző, amely valós időben mutatja a mikrofon jelszintjét, így a felhasználó azonnal látja, ha túl halk vagy túl hangos a környezet.
- **Állapotjelzések**: Vizuálisan mutatja az aktuális fázist (Készenlét, Felvétel..., Feldolgozás..., Tanulás...).

### 🧠 Helyi LoRA Finomhangolás & Javítóablak
- **Interaktív tanulás**: Ha a neurális hálózat félrehallott egy szót, a grafikus felületen a felhasználó beírhatja a helyes szöveget. A „Tanulás” gombra kattintva a szoftver azonnal frissíti a helyi LoRA adapter súlyait.
- **Automatikus biztonsági mentések**: Minden sikeres tanulási ciklus után a szoftver automatikus, időbélyeggel ellátott `.zip` mentést készít a teljes adapter-modellről a `voicetex_backups` mappába, megelőzve az adatvesztést.

### 🏭 Automata Gyár (Batch Feldolgozó)
- **Hosszú fájlok darabolása**: Képes többórás `.wav` hanganyagokat és a hozzájuk tartozó teljes szöveges kéziratokat automatikusan feldolgozni.
- **Intelligens csend-szűrés**: A szoftver automatikusan megkeresi a beszéd szüneteit, és ott szeleteli fel a hangot optimális méretű tanító-mintákra.
- **Whisper-alapú szinkronellenőrzés**: A rendszer a szeletelés után összeveti a vágott hangot a kézirat adott részével. Ha a hibaarány túl nagy, a mintát biztonsági okokból kiszűri, így csak a tökéletes párosok kerülnek a tanító-adatbázisba.

### 🎨 Modernizált Sötét Mód (GUI)
- **Szemkímélő felület**: Teljesen testreszabott, sötét tónusú grafikus felhasználói felület (Tkinter alapokon), amely éjszakai vagy hosszú távú munka során sem fárasztja a szemet.

---

## Technikai felépítés (Fejlesztőknek)

A projekt a Hugging Face `transformers` és a `peft` könyvtáraira épül. A beszédfelismerés magját egy előre tanított OpenAI Whisper modell adja, amelyre egy helyi, magyar nyelvű LoRA adapter réteg épül. 

A projekt fontosabb könyvtárai és szerepük:
- `tanito_adatok/`: Ide kerülnek a felhasználó által jóváhagyott, vagy a Gyár által felszeletelt hang- és szövegpárok.
- `whisper_lora_magyar/`: Az aktuális, aktív LoRA adapter súlyai és konfigurációs fájljai.
- `voicetex_backups/`: Az automatikus zip biztonsági mentések helye.
- `temp_szeletek/`: Ideiglenes munka-könyvtár az automata darabolás során.

---

## Rendszerkövetelmények & Telepítés

A szoftver futtatáshoz Python 3.10 vagy újabb környezet szükséges. A hardverigény a választott Whisper modelltől függ (erősen ajánlott egy Nvidia CUDA-képes videokártya a gyors helyi futtatáshoz és tanításhoz).

### Szükséges könyvtárak telepítése:
Nyiss meg egy parancssort a projekt mappájában, és futtasd az alábbi parancsot:
```bash
pip install torch transformers datasets peft numpy scipy sounddevice pyttsx3 librosa soundfile num2words keyboard