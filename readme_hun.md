# Voicetex Ultra - Push-To-Talk AI Diktáló és Finomhangoló Szoftver

A **Voicetex Ultra** egy professzionális, helyben futtatható, Push-To-Talk (PTT) alapú magyar hangfelismerő és intelligens diktáló alkalmazás. A projekt az OpenAI legmodernebb nyílt forráskódú beszédfelismerő modelljét ötvözi a Low-Rank Adaptation (LoRA) finomhangolási technológiával, lehetővé téve, hogy a szoftver folyamatosan alkalmazkodjon a felhasználó egyedi beszédstílusához, szakszavaihoz és kiejtéséhez.

Remélem, hogy ez a program sokaknak fog örömet okozni, és jelentősen megkönnyíti majd a mindennapi munkát vagy a digitális tartalomgyártást!

\---

## 🚀 Főbb jellemzők és működés

* **Push-To-Talk (PTT) vezérlés:** A rögzítés csak addig tart, amíg a `Ctrl + Win` gombokat nyomva tartod. Elengedéskor a hangfelvétel azonnal lezárul, és kezdetét veszi a feldolgozás.
* **Automatikus vágólap és beillesztés:** A felismert szöveg azonnal a Windows vágólapjára kerül, és a szoftver automatikusan be is injektálja (beilleszti) azt az éppen aktív kurzor pozíciójához (pl. Word, böngésző, jegyzettömb).
* **Élő HUD és VU-meter:** Egy keret nélküli, lebegő indikátor ablak folyamatosan a legfelső rétegen marad, vizuális visszajelzést (élő kivezérlésjelzőt) adva a felvétel állapotáról (Izzó vörös: rögzítés, Sárga: AI feldolgozás, Sötétszürke: Standby).
* **Interaktív tanulási mód:** Ha a gép félrehallott egy szót, a felületen kijavíthatod, a gomb megnyomásával pedig a szoftver a háttérben azonnal megtanulja a javítást.
* **Kötegelt gyári tanítás (Batch Factory):** Lehetőséget biztosít egy már meglévő, hosszú hanganyag (.wav) és a hozzá tartozó pontos kézirat (.txt) automatikus felszeletelésére és tömeges betanítására.

\---

## 🤖 Alkalmazott AI modellek és technológiák

### Milyen LLM/Transformer modellt használ?

A szoftver az **OpenAI Whisper-Large-v3** modelljére épül (`openai/whisper-large-v3`).

* Ez nem egy felhőalapú API, a modell **teljesen helyben, a te számítógépeden fut** (preferáltan egy Nvidia videokártya CUDA magjain, de CPU-n is működik).
* A Whisper-Large-v3 egy 1,5 milliárd paraméteres encoder-decoder architektúrájú Transformer modell, amelyet több millió órányi többnyelvű hanganyagon tanítottak, és kiemelkedő pontossággal kezeli a magyar nyelvet.

### Hol találhatóak a modellek és a mentések?

* **Alapmodell:** Az első indításkor a Hugging Face Transformers könyvtára automatikusan letölti az OpenAI hivatalos szervereiről a modellt a felhasználói fiókod gyorsítótárába (`\~/.cache/huggingface/hub/`).
* **Saját LoRA adapterek:** A finomhangolás során keletkező egyedi súlyok és módosítások a projekt gyökérmappájában, a `./whisper\_lora\_magyar` könyvtárban tárolódnak. Indításkor a szoftver ellenőrzi ezt a mappát, és ha talál egyedi adaptert, automatikusan rátölti azt az alapmodellre.

### Milyen matematikai és AI technikákat használ a szoftver?

1. **LoRA (Low-Rank Adaptation):** Nem a teljes 1,5 milliárd paraméteres modellt tanítjuk újra (ami óriási erőforrást igényelne), hanem csak kisméretű, alacsony rangú súlymátrixokat illesztünk be a Transformer figyelem (Attention - `q\_proj`, `v\_proj`) rétegeibe. Ez gyors és stabil tanulást biztosít akár pár másodperces hangmintákból is.
2. **Hang-szeletelés (Librosa split):** A kötegelt tanításnál a Librosa RMS (Root-Mean-Square) energia-alapú algoritmusával automatikusan detektálja a csendeket, és a mondathatárok mentén vágja fel a nagy hangfájlt.
3. **Szöveg-normalizálás és Num2Words:** A hanganyag és a kézirat összehasonlítása előtt a szoftver eltávolítja az írásjeleket, kisbetűssé alakít mindent, és a számokat (pl. "2026") betűkké konvertálja ("húszezer-huszonhat"), hogy a szinkronitás-ellenőrzés tökéletes legyen.
4. **Difflib Sequence Matching:** A szoftver matematikai hasonlósági arányt számol a gép által tippelt és a valós kézirat között. Ha az egyezőség 40% alatti, a szoftver biztonsági okokból elveti a mintát, megakadályozva a hibás adatok betanulását.

\---

## 💻 Hogyan kell használni?

### Előfeltételek és telepítés

Szükséged lesz egy Python 3.10+ környezetre és a szükséges könyvtárakra. Telepítés:

```bash
pip install torch torchvision torchaudio --index-url \[https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install transformers peft datasets sounddevice scipy librosa soundfile keyboard pyttsx3 num2words


A program indítása

Futtasd a szkriptet a parancssorból:



Bash

python 2026-05-23\_06-30\_ptt\_stable\_hud\_voicetex.py

Mindennapi használat (Diktálás)

Válaszd ki az aktív mikrofonodat a legördülő menüből.



Kattints bele abba a szoftverbe (pl. Word, Notepad, Chat ablak), ahová írni szeretnél.



Nyomd le és TARTSD NYOMVA a Ctrl + Win gombokat. A jobb alsó sarokban a HUD ablak vörösre vált.



Beszélj folyamatosan.



ENGEDD EL a gombokat. A HUD sárgára vált, az AI lefordítja a hangot, és a szöveg máris megjelenik a kurzorodnál!



⚖️ Jogi nyilatkozat és licencek (Disclaimer)

Ez a projekt a nyílt forráskódú (Open Source) közösség elveit követve, kizárólag legális és szabadon felhasználható technológiákra épül. A fejlesztés során kiemelt figyelmet fordítottam a licencek betartására:



OpenAI Whisper-Large-v3: MIT Licenc (Szabadon felhasználható, módosítható).



Hugging Face PEFT \& Transformers: Apache 2.0 Licenc (Megengedi a szabad magán- és kereskedelmi használatot).



Python modulok: MIT, BSD és Apache 2.0 licencek alapján beépítve.



A legjobb tudomásom és alapos utánajárásom szerint a forráskód semmilyen szerzői jogot vagy szellemi tulajdont nem sért.



📞 Hibabejelentési és eljárási nyilatkozat (Take-Down Policy)

Ha Ön jogtulajdonosként mégis bármilyen licenc-ütközést vagy aggályt tapasztal a kódban, kérlek, NE indítson azonnal hivatalos eljárást. Nyiss egy "Issue"-t itt a GitHubon, és garantálom, hogy a bejelentést azonnal kivizsgálom, és a jelzett problémát a lehető leghamarabb javítom vagy eltávolítom a projektből.



💬 Vélemények, ötletek és fejlesztés – Várom a visszajelzésed!

Nagyon örülnék neki, ha elmondanád a véleményed, amennyiben használod a szoftvert! Kifejezetten várom a visszajelzéseket az alábbiakban:



Hibák bejelentése (Bug reports): Ha bármilyen anomáliát, fagyást vagy hibát tapasztalsz.



Ötletek és javaslatok: Milyen funkciókkal lehetne még kiegészíteni? (Pl. egyedi írásjel-parancsok, új gyorsbillentyűk, stb.)



Fejlesztési irányok: Szerinted hogyan lehetne még jobbá, gyorsabbá vagy hatékonyabbá tenni a kódot?



Kérlek, oszd meg a gondolataidat a GitHub Issues vagy Discussions fül alatt! Építsük és fejlesszük ezt a projektet közösen!

