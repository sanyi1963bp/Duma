import os
import sys
import time
import queue
import threading
import re
import numpy as np
import scipy.io.wavfile as wav
import torch
import tkinter as tk
from tkinter import messagebox, ttk, filedialog

try:
    import sounddevice as sd
    MIKROFON_ELERHETO = True
except Exception:
    MIKROFON_ELERHETO = False

try:
    import pyttsx3
    TTS_ELERHETO = True
except Exception:
    TTS_ELERHETO = False

try:
    import librosa
    import soundfile as sf
    DARABOLO_ELERHETO = True
except Exception:
    DARABOLO_ELERHETO = False

try:
    from num2words import num2words
    NUM2WORDS_ELERHETO = True
except Exception:
    NUM2WORDS_ELERHETO = False

try:
    import keyboard
    KEYBOARD_ELERHETO = True
except Exception:
    KEYBOARD_ELERHETO = False

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel

# --- BEÁLLÍTÁSOK ---
MINTAVETELI_FREKVENCIA = 16000
MODEL_NAME = "openai/whisper-large-v3"
LORA_OUTPUT_DIR = "./whisper_lora_magyar"
DATASET_DIR = "./tanito_adatok"
GLOBAL_HOTKEY = "ctrl+win"  # A Push-to-Talk gyorsbillentyű kombináció

def szoveg_normalizalas(szoveg):
    szoveg = szoveg.lower().strip()
    szoveg = re.sub(r'[.,;:!?\-"()“”„”«»—_]', ' ', szoveg)
    if NUM2WORDS_ELERHETO:
        szamok = re.findall(r'\d+', szoveg)
        for szam in sorted(szamok, key=len, reverse=True):
            try:
                betus_szam = num2words(int(szam), lang='hu')
                szoveg = szoveg.replace(szam, " " + betus_szam + " ")
            except Exception:
                pass
    szoveg = re.sub(r'\s+', ' ', szoveg).strip()
    return szoveg

def hasonlosag_arány(s1, s2):
    s1, s2 = szoveg_normalizalas(s1), szoveg_normalizalas(s2)
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    import difflib
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def gép_beszél(szöveg):
    if TTS_ELERHETO:
        def tts_thread():
            try:
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                for voice in voices:
                    if "hungary" in voice.name.lower() or "hu" in voice.id.lower():
                        engine.setProperty('voice', voice.id)
                        break
                engine.setProperty('rate', 165)
                engine.say(szöveg)
                engine.runAndWait()
            except Exception as e:
                print(f"TTS hiba: {e}")
        threading.Thread(target=tts_thread, daemon=True).start()

def hatter_tanitas_process(hang_utvonal, javitott_szoveg, device):
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="hungarian", task="transcribe")
    sr, audio_np = wav.read(hang_utvonal)
    audio_np = audio_np.astype(np.float32) / 32767.0
    
    input_features = processor(audio_np, sampling_rate=MINTAVETELI_FREKVENCIA).input_features[0]
    labels = processor.tokenizer(javitott_szoveg).input_ids
    
    adat = {"input_features": input_features.tolist(), "labels": labels}
    dataset = Dataset.from_list([adat])
    
    alap_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    if os.path.exists(os.path.join(LORA_OUTPUT_DIR, "adapter_config.json")):
        model_to_train = PeftModel.from_pretrained(alap_model, LORA_OUTPUT_DIR, is_trainable=True).to(device)
    else:
        config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
        model_to_train = get_peft_model(alap_model, config).to(device)

    def egyedi_data_collator(features):
        batch_features = [{"input_features": torch.tensor(f["input_features"])} for f in features]
        batch_labels = [{"input_ids": torch.tensor(f["labels"])} for f in features]
        padded_features = processor.feature_extractor.pad(batch_features, return_tensors="pt")
        padded_labels = processor.tokenizer.pad(batch_labels, return_tensors="pt")
        labels_tensor = padded_labels["input_ids"].masked_fill(padded_labels.attention_mask.ne(1), -100)
        return {"input_features": padded_features["input_features"], "labels": labels_tensor}

    training_args = Seq2SeqTrainingArguments(
        output_dir=LORA_OUTPUT_DIR, per_device_train_batch_size=1, learning_rate=2e-4,
        max_steps=5, fp16=torch.cuda.is_available(), remove_unused_columns=False,
        label_names=["labels"], report_to="none"
    )

    trainer = Seq2SeqTrainer(
        args=training_args, model=model_to_train, train_dataset=dataset,
        data_collator=egyedi_data_collator, tokenizer=processor.feature_extractor,
    )
    trainer.train()
    model_to_train.save_pretrained(LORA_OUTPUT_DIR)
    del model_to_train
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class VoicetexApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voicetex Gemini Factory Ultra - Push-To-Talk Edition")
        self.root.geometry("780x750")
        self.root.minsize(700, 650)
        
        self.hang_sor = queue.Queue()
        self.is_recording = False
        self.hang_adatok = []
        self.utolso_hang_utvonal = ""
        self.record_start_time = 0
        self.stop_factory_requested = False
        self.minden_eszkoz_lista = []
        self.aktualis_vu_szint = 0.0
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab1, text="🎤 Interaktív Diktálás")
        self.notebook.add(self.tab2, text="🤖 Automata Kötegelt Tanítás")
        
        self.create_diktalo_widgets()
        self.create_factory_widgets()
        self.create_overlay_window()
        
        self.status_bar = ttk.Label(self.root, text="Modell inicializálása... Kérlek várj...", relief="sunken", anchor="w", padding=5)
        self.status_bar.pack(fill="x", side="bottom")
        
        threading.Thread(target=self.init_model, daemon=True).start()

    def create_diktalo_widgets(self):
        control_frame = ttk.LabelFrame(self.tab1, text=" Vezérlés és Push-to-Talk Beállítások ", padding=10)
        control_frame.pack(fill="x", padx=15, pady=10)
        
        dev_label_frame = ttk.Frame(control_frame)
        dev_label_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(dev_label_frame, text="Aktív Bemenet (Mikrofon):", font=("Helvetica", 9, "bold")).pack(side="left", padx=5)
        
        self.device_combo = ttk.Combobox(dev_label_frame, state="readonly", width=50)
        self.device_combo.pack(side="left", padx=5, fill="x", expand=True)
        
        btn_sub_frame = ttk.Frame(control_frame)
        btn_sub_frame.pack(fill="x")
        
        self.record_btn = ttk.Button(btn_sub_frame, text="🎤 Tartsd nyomva a felvételhez", command=self.toggle_recording, state="disabled")
        self.record_btn.pack(side="left", padx=5)
        
        self.time_label = ttk.Label(btn_sub_frame, text="Időtartam: 0.0 mp", font=("Helvetica", 10))
        self.time_label.pack(side="left", padx=15)
        
        hotkey_msg = f"💡 Push-to-Talk: Tartsd nyomva a [CTRL + WIN] kombinációt!"
        self.hotkey_label = ttk.Label(btn_sub_frame, text=hotkey_msg, font=("Helvetica", 9, "bold"), foreground="#008800")
        self.hotkey_label.pack(side="right", padx=5)
        
        vu_frame = ttk.LabelFrame(self.tab1, text=" Élő Mikrofon Kivezérlésjelző (VU) ", padding=10)
        vu_frame.pack(fill="x", padx=15, pady=5)
        
        self.vu_meter = ttk.Progressbar(vu_frame, orient="horizontal", mode="determinate", length=100)
        self.vu_meter.pack(fill="x", expand=True, pady=2)
        
        style = ttk.Style()
        style.configure("VU.Horizontal.TProgressbar", foreground='#00ff00', background='#00ff00')
        self.vu_meter.config(style="VU.Horizontal.TProgressbar")

        text_frame = ttk.Frame(self.tab1, padding=10)
        text_frame.pack(fill="both", expand=True, padx=15, pady=5)
        
        ttk.Label(text_frame, text="Amit a gép hallott (Automatikusan a vágólapra másolva és beillesztve!):", font=("Helvetica", 10, "bold"), foreground="#0066cc").pack(anchor="w")
        self.ai_text = tk.Text(text_frame, height=4, bg="#fbfbfb", font=("Helvetica", 11), wrap="word")
        self.ai_text.pack(fill="x", pady=5)
        self.ai_text.config(state="disabled")
        
        ttk.Label(text_frame, text="Ide írd a javítást (Ebből tanul a gép):", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(10, 0))
        self.user_text = tk.Text(text_frame, height=6, font=("Helvetica", 11), wrap="word")
        self.user_text.pack(fill="both", expand=True, pady=5)
        
        button_frame = ttk.Frame(self.tab1, padding=5)
        button_frame.pack(fill="x", padx=15, pady=10)
        
        self.train_btn = ttk.Button(button_frame, text="💾 Összehasonlítás és Tanulás", command=self.start_learning, state="disabled")
        self.train_btn.pack(side="right", padx=5)

    def create_factory_widgets(self):
        factory_frame = ttk.LabelFrame(self.tab2, text=" Fájlok Betöltése ", padding=15)
        factory_frame.pack(fill="x", padx=15, pady=10)
        
        ttk.Label(factory_frame, text="1. Hosszú tiszta hanganyag (.wav):", font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        self.wave_path_entry = ttk.Entry(factory_frame, width=45)
        self.wave_path_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(factory_frame, text="Tallózás...", command=self.browse_wave).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(factory_frame, text="2. Teljes pontos kézirat (.txt):", font=("Helvetica", 10, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        self.txt_path_entry = ttk.Entry(factory_frame, width=45)
        self.txt_path_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(factory_frame, text="Tallózás...", command=self.browse_txt).grid(row=1, column=2, padx=5, pady=5)
        
        process_frame = ttk.Frame(self.tab2, padding=10)
        process_frame.pack(fill="both", expand=True, padx=15, pady=5)
        
        self.factory_log = tk.Text(process_frame, height=12, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 10))
        self.factory_log.pack(fill="both", expand=True, pady=5)
        
        self.factory_progress = ttk.Progressbar(process_frame, mode="determinate")
        self.factory_progress.pack(fill="x", pady=5)
        
        btn_action_frame = ttk.Frame(process_frame)
        btn_action_frame.pack(fill="x", pady=5)
        
        self.stop_factory_btn = ttk.Button(btn_action_frame, text="🛑 Kényszerített Leállítás", command=self.request_factory_stop, state="disabled")
        self.stop_factory_btn.pack(side="left")
        
        self.start_factory_btn = ttk.Button(btn_action_frame, text="🚀 Automata Kötegelt Tanulás Indítása", command=self.start_batch_processing, state="disabled")
        self.start_factory_btn.pack(side="right")

    def create_overlay_window(self):
        self.overlay = tk.Toplevel(self.root)
        self.overlay.title("Voicetex HUD")
        self.overlay.geometry("200x45+10+10")
        self.overlay.attributes("-topmost", True)
        self.overlay.overrideredirect(True)
        self.overlay.attributes("-alpha", 0.85)
        self.overlay.configure(bg="#222222")
        
        self.overlay_label = tk.Label(self.overlay, text="🎙️ PUSH-TO-TALK KÉSZ", font=("Helvetica", 9, "bold"), fg="#00ff00", bg="#222222")
        self.overlay_label.pack(fill="x", pady=2)
        
        self.overlay_vu = ttk.Progressbar(self.overlay, orient="horizontal", mode="determinate", length=180, style="VU.Horizontal.TProgressbar")
        self.overlay_vu.pack(padx=10, fill="x")
        
        self.overlay.bind("<Button-1>", self.start_drag)
        self.overlay.bind("<B1-Motion>", self.drag_motion)
        self.overlay.withdraw()

    def start_drag(self, event):
        self.drag_x = event.x
        self.drag_y = event.y

    def drag_motion(self, event):
        x = self.overlay.winfo_x() - self.drag_x + event.x
        y = self.overlay.winfo_y() - self.drag_y + event.y
        self.overlay.geometry(f"+{x}+{y}")

    def browse_wave(self):
        fajl = filedialog.askopenfilename(filetypes=[("WAV hangfájl", "*.wav")])
        if fajl:
            self.wave_path_entry.delete(0, tk.END)
            self.wave_path_entry.insert(0, fajl)

    def browse_txt(self):
        fajl = filedialog.askopenfilename(filetypes=[("Szövegfájl", "*.txt")])
        if fajl:
            self.txt_path_entry.delete(0, tk.END)
            self.txt_path_entry.insert(0, fajl)

    def log_to_factory(self, text, color=None):
        self.root.after(0, lambda: self._safe_log(text, color))
        
    def _safe_log(self, text, color):
        self.factory_log.insert(tk.END, text + "\n")
        self.factory_log.see(tk.END)

    def request_factory_stop(self):
        self.stop_factory_requested = True
        self.log_to_factory("\n🛑 LEÁLLÍTÁS KÉRVE! A gyár biztonságosan megáll...")
        self.stop_factory_btn.config(state="disabled")

    def start_batch_processing(self):
        wave_fajl = self.wave_path_entry.get().strip()
        txt_fajl = self.txt_path_entry.get().strip()
        
        if not wave_fajl or not txt_fajl:
            messagebox.showwarning("Figyelem", "Kérlek add meg mind a két fájl elérési útját!")
            return
            
        self.start_factory_btn.config(state="disabled")
        self.stop_factory_btn.config(state="normal")
        self.record_btn.config(state="disabled")
        self.factory_log.delete("1.0", tk.END)
        self.stop_factory_requested = False
        
        threading.Thread(target=self.batch_processing_thread, args=(wave_fajl, txt_fajl,), daemon=True).start()

    def batch_processing_thread(self, wave_path, txt_path):
        try:
            self.log_to_factory("📖 Kézirat beolvasása...")
            with open(txt_path, "r", encoding="utf-8") as f:
                teljes_szoveg = f.read().replace("\n", " ").strip()
            
            mondatok = [m.strip() for m in teljes_szoveg.split(".") if m.strip()]
            self.log_to_factory(f"✨ Talált mondatok száma: {len(mondatok)}")
            
            self.log_to_factory("🎵 Hanganyag betöltése és szeletelése...")
            y, sr = librosa.load(wave_path, sr=MINTAVETELI_FREKVENCIA)
            intervals = librosa.effects.split(y, top_db=26, frame_length=2048, hop_length=512)
            
            self.log_to_factory(f"✂️ Szeletek száma: {len(intervals)}")
            
            feldolgozando = min(len(mondatok), len(intervals))
            self.root.after(0, lambda: self.factory_progress.config(max=feldolgozando, value=0))
            
            temp_dir = "./temp_szeletek"
            os.makedirs(temp_dir, exist_ok=True)

            for i in range(feldolgozando):
                if self.stop_factory_requested:
                    self.log_to_factory("\n🛑 Gyár leállítva. Az eddigi adatok elmentve!")
                    break
                    
                aktualis_mondat = mondatok[i] + "."
                self.log_to_factory(f"\n[Mondat {i+1}/{feldolgozando}]")
                
                start_idx, end_idx = intervals[i]
                szelet = y[start_idx:end_idx]
                
                rms = np.sqrt(np.mean(szelet**2))
                if rms < 0.005:
                    self.log_to_factory("🤫 Túl halk szelet (csend). Kihagyjuk a hallucinációk elkerülésére.")
                    self.root.after(0, lambda v=i+1: self.factory_progress.config(value=v))
                    continue

                szelet_utvonal = os.path.join(temp_dir, f"szelet_{i}.wav")
                sf.write(szelet_utvonal, szelet, MINTAVETELI_FREKVENCIA)
                
                input_features = self.processor(szelet, sampling_rate=MINTAVETELI_FREKVENCIA, return_tensors="pt").input_features.to(self.device)
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        input_features, language="hungarian", task="transcribe",
                        max_new_tokens=256, num_beams=3, do_sample=False
                    )
                gép_tippje = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                if any(bad in gép_tippje.lower() for bad in ["amara", "felirat", "subtitle", "közösség"]):
                    gép_tippje = ""

                hasonlosag = hasonlosag_arány(gép_tippje, aktualis_mondat)
                self.log_to_factory(f"🔍 Tipp: '{gép_tippje}'")
                self.log_to_factory(f"📊 Szinkron egyezőség: {hasonlosag*100:.1f}%")
                
                if hasonlosag < 0.40:
                    self.log_to_factory("⚠️ Szinkronhiba veszély! Kihagyjuk.", "orange")
                    if os.path.exists(szelet_utvonal):
                        os.remove(szelet_utvonal)
                    self.root.after(0, lambda v=i+1: self.factory_progress.config(value=v))
                    continue
                
                self.log_to_factory("🧠 Tanítás és azonnali mentési pont írása...")
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                hatter_tanitas_process(szelet_utvonal, aktualis_mondat, self.device)
                self.load_active_model()
                
                if os.path.exists(szelet_utvonal):
                    os.remove(szelet_utvonal)
                    
                self.root.after(0, lambda v=i+1: self.factory_progress.config(value=v))
                self.log_to_factory("✅ Mentési pont sikeresen lemezre írva!")
            
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
                
            self.log_to_factory("\n🏁 FOLYAMAT BEFEJEZŐDÖTT!")
            gép_beszél("A csúcsmodell kötegelt feldolgozása befejeződött.")
            self.root.after(0, lambda: messagebox.showinfo("Kész", "A Whisper-Large-v3 tanítása lefutott!"))
            
        except Exception as e:
            self.log_to_factory(f"❌ Hiba: {e}")
            self.load_active_model()
            
        self.root.after(0, lambda: self.start_factory_btn.config(state="normal"))
        self.root.after(0, lambda: self.stop_factory_btn.config(state="disabled"))
        self.root.after(0, lambda: self.record_btn.config(state="normal"))

    def init_model(self):
        try:
            self.processor = WhisperProcessor.from_pretrained(MODEL_NAME)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.load_active_model()
            self.root.after(0, self.enable_ui_after_load)
        except Exception as e:
            self.root.after(0, lambda m=str(e): self.show_error_popup(m))

    def show_error_popup(self, msg):
        self.status_bar.config(text="❌ Hiba történt.")
        messagebox.showerror("Hiba", f"Nem sikerült elindítani a modellt.\n\nRészletek:\n{msg}")

    def load_active_model(self):
        if os.path.exists(os.path.join(LORA_OUTPUT_DIR, "adapter_config.json")):
            self.log_status("💡 LoRA mentési pont betöltése...")
            alap_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(self.device)
            self.model = PeftModel.from_pretrained(alap_model, LORA_OUTPUT_DIR).to(self.device)
        else:
            self.log_status("🌱 Gyári OpenAI Whisper-Large-v3 betöltése...")
            self.model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(self.device)

    def enable_ui_after_load(self):
        self.record_btn.config(state="normal")
        self.start_factory_btn.config(state="normal")
        self.frissit_vu_grafika()
        self.feltolt_hangeszkozok()
        
        self.overlay.deiconify()
        
        # INTELLIGENS PUSH-TO-TALK BILLENTYŰZET-ESEMÉNYEK KÖTÉSE
        if KEYBOARD_ELERHETO:
            try:
                # Amikor lenyomja a gombot -> Elindul a rögzítés
                keyboard.on_press_key("win", self.hotkey_down_check)
                # Amikor felengedi a gombot -> Leáll a rögzítés
                keyboard.on_release_key("win", self.hotkey_up_check)
            except Exception as e:
                print(f"Push-to-Talk regisztrációs hiba: {e}")

    # Ellenőrizzük, hogy a Win gomb mellett a Ctrl is le van-e nyomva
    def hotkey_down_check(self, event):
        if keyboard.is_pressed("ctrl") and not self.is_recording:
            self.root.after(0, self.start_recording_process)

    def hotkey_up_check(self, event):
        if self.is_recording:
            self.root.after(0, self.stop_recording_process)

    def feltolt_hangeszkozok(self):
        if not MIKROFON_ELERHETO:
            self.device_combo.config(values=["Nincs elérhető hangillesztő!"])
            self.device_combo.current(0)
            return

        try:
            eszkozok = sd.query_devices()
            bemeneti_nevek = []
            self.minden_eszkoz_lista = []
            
            preferalt_index = 0
            szamlalo = 0
            
            for i, dev in enumerate(eszkozok):
                if dev['max_input_channels'] > 0:
                    nev = f"{i}: {dev['name']} ({int(dev['max_input_channels'])} ch)"
                    bemeneti_nevek.append(nev)
                    self.minden_eszkoz_lista.append(i)
                    if "96" in dev['name'].lower() or "audiobox" in dev['name'].lower() or "usb" in dev['name'].lower():
                        preferalt_index = szamlalo
                    szamlalo += 1
            
            if bemeneti_nevek:
                self.device_combo.config(values=bemeneti_nevek)
                self.device_combo.current(preferalt_index)
                valasztott = self.device_combo.get()
                self.status_bar.config(text=f"🚀 PTT Mód Aktív! Tartsd nyomva a [CTRL+WIN]-t! Eszköz: {valasztott}")
            else:
                self.device_combo.config(values=["Nem található bemeneti eszköz!"])
                self.device_combo.current(0)
        except Exception as e:
            self.status_bar.config(text=f"⚠️ Eszköz-pásztázási hiba: {e}")

    def log_status(self, text):
        self.root.after(0, lambda: self.status_bar.config(text=text))

    # Manuális kattintás szimulációja a UI gombhoz (ha mégis ráklikkelnél)
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording_process()
        else:
            self.stop_recording_process()

    # --- RÖGZÍTÉS CÉLZOTT INDÍTÁSA LENYOMÁSKOR ---
    def start_recording_process(self):
        if self.is_recording: return
        self.is_recording = True
        self.record_btn.config(text="🛑 Engedd el a leállításhoz")
        self.device_combo.config(state="disabled")
        self.overlay_label.config(text="🎙️ RÖGZÍTÉS (PTT)...", fg="#ff3333")
        
        self.hang_adatok = []
        self.record_start_time = time.time()
        self.update_timer()
        threading.Thread(target=self.record_audio_thread, daemon=True).start()

    # --- RÖGZÍTÉS CÉLZOTT LEÁLLÍTÁSA FELENGEDÉSKOR ---
    def stop_recording_process(self):
        if not self.is_recording: return
        self.is_recording = False
        self.record_btn.config(state="disabled", text="⏳ Feldolgozás...")
        self.overlay_label.config(text="⏳ WHISPER FORDÍT...", fg="#ffff00")

    def update_timer(self):
        if self.is_recording:
            eltelt = time.time() - self.record_start_time
            self.time_label.config(text=f"Időtartam: {eltelt:.1f} mp")
            self.root.after(100, self.update_timer)

    def frissit_vu_grafika(self):
        if self.is_recording:
            ertek = int(self.aktualis_vu_szint * 350)
            if ertek > 100: ertek = 100
            self.vu_meter.config(value=ertek)
            self.overlay_vu.config(value=ertek)
        else:
            self.vu_meter.config(value=0)
            self.overlay_vu.config(value=0)
        self.root.after(30, self.frissit_vu_grafika)

    def record_audio_thread(self):
        fajlnev = f"minta_{int(time.time())}.wav"
        os.makedirs(DATASET_DIR, exist_ok=True)
        self.utolso_hang_utvonal = os.path.join(DATASET_DIR, fajlnev)
        
        try:
            valasztott_idx = self.device_combo.current()
            if valasztott_idx >= 0 and len(self.minden_eszkoz_lista) > valasztott_idx:
                input_device = self.minden_eszkoz_lista[valasztott_idx]
                eszkoz_adat = sd.query_devices(input_device)
                max_csatorna = int(eszkoz_adat['max_input_channels'])
            else:
                input_device = None
                max_csatorna = 1

            def callback(indata, frames, time, status):
                if self.is_recording:
                    self.hang_sor.put(indata.copy())
                    if indata.size > 0:
                        rms = np.sqrt(np.mean(indata**2))
                        self.aktualis_vu_szint = rms
                
            if MIKROFON_ELERHETO and input_device is not None:
                with sd.InputStream(samplerate=MINTAVETELI_FREKVENCIA, channels=max_csatorna, device=input_device, callback=callback):
                    while self.is_recording or not self.hang_sor.empty():
                        try:
                            data = self.hang_sor.get(timeout=0.05)
                            self.hang_adatok.append(data)
                        except queue.Empty:
                            continue
            else:
                while self.is_recording:
                    time.sleep(0.1)
                kamu_hang = np.zeros((MINTAVETELI_FREKVENCIA * 3, 1), dtype=np.float32)
                self.hang_adatok.append(kamu_hang)

            if len(self.hang_adatok) > 0:
                teljes_hang = np.concatenate(self.hang_adatok, axis=0)
                
                if teljes_hang.ndim > 1 and teljes_hang.shape[1] > 1:
                    csatorna_energia = [np.max(np.abs(teljes_hang[:, ch])) for ch in range(teljes_hang.shape[1])]
                    legaktivabb_ch = np.argmax(csatorna_energia)
                    mono_hang = teljes_hang[:, legaktivabb_ch]
                else:
                    mono_hang = teljes_hang.flatten()

                wav.write(self.utolso_hang_utvonal, MINTAVETELI_FREKVENCIA, (mono_hang * 32767).astype(np.int16))
                self.run_inference()
            else:
                self.root.after(0, self.reset_record_button)

        except Exception as e:
            self.root.after(0, self.reset_record_button)

    def run_inference(self):
        try:
            sr, audio_np = wav.read(self.utolso_hang_utvonal)
            audio_np = audio_np.astype(np.float32) / 32767.0
            
            rms_ertek = np.sqrt(np.mean(audio_np**2))
            if rms_ertek < 0.0015:
                self.root.after(0, lambda: self.display_inference_result("[Süket csendet észleltem]"))
                return

            input_features = self.processor(audio_np, sampling_rate=MINTAVETELI_FREKVENCIA, return_tensors="pt").input_features.to(self.device)
            
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features, language="hungarian", task="transcribe",
                    max_new_tokens=256, num_beams=3, do_sample=False
                )
                
            text_result = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            if any(bad in text_result.lower() for bad in ["amara", "felirat", "subtitle", "közösség"]):
                text_result = ""

            # Garantált vágólapra tétel
            self.root.clipboard_clear()
            self.root.clipboard_append(text_result)
            self.root.update()
            
            if KEYBOARD_ELERHETO and text_result.strip():
                time.sleep(0.12)
                keyboard.send("ctrl+v")

            self.root.after(0, lambda: self.display_inference_result(text_result))
        except Exception as e:
            self.root.after(0, lambda m=str(e): self.display_error_in_bar(m))

    def display_error_in_bar(self, msg):
        self.status_bar.config(text=f"❌ Hiba: {msg[:50]}...")
        self.reset_record_button()

    def display_inference_result(self, text):
        self.ai_text.config(state="normal")
        self.ai_text.delete("1.0", tk.END)
        self.ai_text.insert("1.0", text)
        self.ai_text.config(state="disabled")
        
        self.user_text.delete("1.0", tk.END)
        self.user_text.insert("1.0", text)
        
        self.reset_record_button()
        self.train_btn.config(state="normal")
        self.log_status(f"📋 Szöveg a vágólapon és beinjektálva!")

    def reset_record_button(self):
        self.is_recording = False
        self.aktualis_vu_szint = 0.0
        self.record_btn.config(state="normal", text="🎤 Tartsd nyomva a felvételhez")
        self.device_combo.config(state="readonly")
        self.overlay_label.config(text="🎙️ PUSH-TO-TALK KÉSZ", fg="#00ff00")

    def start_learning(self):
        javitott_szoveg = self.user_text.get("1.0", tk.END).strip()
        if not javitott_szoveg:
            messagebox.showwarning("Figyelem", "Üres szövegből nem tudok tanulni!")
            return
            
        self.train_btn.config(state="disabled")
        self.record_btn.config(state="disabled")
        self.log_status("🧠 Modell finomhangolása... Kérlek várj!")
        threading.Thread(target=self.run_train_wrapper, args=(javitott_szoveg,), daemon=True).start()

    def run_train_wrapper(self, javitott_szoveg):
        try:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            hatter_tanitas_process(self.utolso_hang_utvonal, javitott_szoveg, self.device)
            self.load_active_model()
            self.log_status("💾 Sikeres tanulás! A modell frissült.")
            gép_beszél("Sikeresen megtanultam a javítást!")
            self.root.after(0, lambda: messagebox.showinfo("Siker", "Megtanultam!"))
        except Exception as e:
            self.log_status(f"Hiba a tanulás során: {e}")
            self.load_active_model()
            
        self.root.after(0, lambda: self.record_btn.config(state="normal"))


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('vista' if 'vista' in style.theme_names() else 'default')
    app = VoicetexApp(root)
    root.mainloop()