#!/usr/bin/env python
"""
Transcribe a single-speaker WAV and highlight fillers.

Usage:
    python transcribe.py  interviewee.wav
Output:
    interviewee.txt  – transcript with UM / UH / … in UPPER-CASE
"""

# ─── bootstrap: block TensorFlow + old-TBB issues ───────────────────────
import os, sys, types, importlib.machinery
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

class _TFStub(types.ModuleType):                  # fake tiny tf module
    def __init__(self, name):
        super().__init__(name)
        self.__spec__  = importlib.machinery.ModuleSpec(name, loader=None)
        self.__file__  = "<stub>"
        class _Dummy: pass
        self.Tensor = self.Variable = _Dummy
    def __getattr__(self, item):
        mod = _TFStub(f"{self.__name__}.{item}")
        sys.modules[mod.__name__] = mod
        return mod
sys.modules["tensorflow"] = _TFStub("tensorflow")
# ────────────────────────────────────────────────────────────────────────

import pathlib, itertools, re
import soundfile as sf, librosa
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq

MODEL_ID = "openai/whisper-medium.en"          # Medium model better at fillers than small
FILLERS  = {"um","uh","erm","hmm","ah","eh","like","you know","so","well","actually","basically","right","okay"}

proc  = AutoProcessor.from_pretrained(MODEL_ID)
if proc.tokenizer.pad_token_id is None:              # avert padding error
    proc.tokenizer.pad_token_id = proc.tokenizer.eos_token_id

model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID)

# ― ensure alignment_heads so word-timestamps work everywhere ------------
if not getattr(model.generation_config, "alignment_heads", None):
    # default heads for Whisper-small (works for other sizes too)
    model.generation_config.alignment_heads = [[2, 2], [6, 2], [9, 2], [12, 2]]

asr = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=proc.tokenizer,
    feature_extractor=proc.feature_extractor,
    return_timestamps="word",
    chunk_length_s=15,  # Shorter chunks for better filler detection
    device="cpu",
)

def mmss(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"

# ─── CLI & resample to 16 kHz ───────────────────────────────────────────
if len(sys.argv) < 2:
    sys.exit("Usage:  python transcribe.py  interviewee.wav")

wav_path        = pathlib.Path(sys.argv[1]).expanduser()
audio, orig_sr  = sf.read(wav_path)
if orig_sr != 16_000:
    audio = librosa.resample(audio, orig_sr, 16_000)
tmp = wav_path.with_suffix(".16k.tmp.wav")
sf.write(tmp, audio, 16_000)

result = asr(str(tmp), batch_size=4)          # Smaller batch for more detailed processing
words  = result["chunks"]                     # list of {text, timestamp}

# Enhanced filler detection
def detect_short_words(ws):
    """Detect potential fillers from very short word segments"""
    fillers_found = []
    for w in ws:
        if "timestamp" in w and len(w["timestamp"]) == 2:
            duration = w["timestamp"][1] - w["timestamp"][0]
            text = w["text"].strip().lower()
            
            # Very short words (< 0.8s) with common filler patterns
            if duration < 0.8 and (
                len(text) <= 3 or 
                text in FILLERS or
                any(f in text for f in ["um", "uh", "ah", "eh"])
            ):
                fillers_found.append((w["timestamp"][0], f"[{text.upper()}]"))
    
    return fillers_found

detected_fillers = detect_short_words(words)

# group ~5-s blocks and UPPER-CASE fillers
def grouped(ws):
    for _, g in itertools.groupby(ws, key=lambda w: int(w["timestamp"][0]//5)):
        g = list(g)
        start = g[0]["timestamp"][0]
        text  = " ".join(w["text"] for w in g)

        pat   = r"\b(" + "|".join(re.escape(f) for f in FILLERS) + r")\b"
        text  = re.sub(pat, lambda m: m.group(1).upper(), text, flags=re.I)
        yield start, text

out_file = wav_path.with_suffix(".txt")
with out_file.open("w") as fp:
    for st, line in grouped(words):
        fp.write(f'At {mmss(st)}, Speaker: "{line}"\n')

print("✓ transcript saved →", out_file.name)