#!/usr/bin/env python3
"""
Verbatim transcription (filler-words kept) for a *single-speaker* WAV.

Usage
-----
python transcribe_test.py  interviewee.wav

Output
------
interviewee.txt   – one line per ≈5 s block, with UM/UH/HM upper-cased.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ────── 1. bootstrap *before* any third-party import ───────────────────
import sys, types, importlib.machinery, os
os.environ["TRANSFORMERS_NO_TF"] = "1"          # transformers: skip TF
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class _TFStub(types.ModuleType):                # tiny fake TensorFlow
    def __init__(self, name):
        super().__init__(name)
        self.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        self.__file__ = "<stub>"
        class _Dummy: pass
        self.Tensor = self.Variable = _Dummy
    def __getattr__(self, item):
        full = f"{self.__name__}.{item}"
        mod  = _TFStub(full)
        sys.modules[full] = mod
        return mod

sys.modules["tensorflow"] = _TFStub("tensorflow")

# ────── 2. standard libs & third-party imports ─────────────────────────
import pathlib, itertools, re
import soundfile as sf, librosa
import numpy as np

# ────── 2a. fast ASR imports ──────────────────────────────────────────
from faster_whisper import WhisperModel          # <= fast, filler‑friendly ASR
import torch

# ────── 3. configuration ───────────────────────────────────────────────
MODEL_ID   = "small.en"   # Use local model name instead of full path
FILLERS    = {"um","uh","erm","hmm","ah","eh","like","you know","so","well","actually","basically"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"  # int8 is fastest for CPU, or use "float32" for full accuracy

# ────── 4. helpers ─────────────────────────────────────────────────────
def mmss(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"

def grouped(words, window=5):
    key = lambda w: int(w["timestamp"][0] // window)
    pat = re.compile(rf"\b({'|'.join(map(re.escape, FILLERS))})\b", flags=re.I)
    for _, g in itertools.groupby(words, key=key):
        block = list(g)
        start = block[0]["timestamp"][0]
        txt   = " ".join(w["text"] for w in block)
        txt   = pat.sub(lambda m: m.group(1).upper(), txt)
        yield start, txt

# ────── 5. CLI & audio prep ────────────────────────────────────────────
if len(sys.argv) < 2:
    sys.exit("Usage: python transcribe_test.py  interviewee.wav")

wav_path = pathlib.Path(sys.argv[1]).expanduser()
if not wav_path.exists():
    sys.exit(f"File not found: {wav_path}")

audio, sr = sf.read(wav_path)
if sr != 16_000:
    audio = librosa.resample(audio, sr, 16_000)
    sr = 16_000

# ────── 6. load Faster‑Whisper model ─────────────────────────────────
try:
    model = WhisperModel(MODEL_ID, device=DEVICE, compute_type=COMPUTE_TYPE)
except Exception as e:
    print(f"Failed to load {MODEL_ID}, falling back to base model")
    model = WhisperModel("base.en", device=DEVICE, compute_type="int8")

# ────── 7. transcription with word timestamps ────────────────────────────────────────
segments, _ = model.transcribe(
    audio.astype("float32"),
    beam_size=1,
    vad_filter=False,  # Turn off VAD to capture more speech
    language="en",
    word_timestamps=True,  # Enable word timestamps
    condition_on_previous_text=False,  # Reduce context bias
    suppress_tokens=[-1],  # Don't suppress any tokens
    initial_prompt="Include all speech sounds including um, uh, like, you know, so, well, actually, basically, hmm, ah, eh, erm."
)

# ────── 8. enhanced filler detection ───────────────────────────────────────
def detect_audio_fillers(audio_segment, sr, start_time, duration=0.5):
    """Detect potential fillers using audio characteristics"""
    if len(audio_segment) < sr * 0.1:  # Skip very short segments
        return []
    
    # Calculate spectral features
    stft = librosa.stft(audio_segment, n_fft=512, hop_length=256)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=np.abs(stft)))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_segment))
    
    # Filler characteristics: lower spectral centroid, higher ZCR
    if spectral_centroid < 2000 and zero_crossing_rate > 0.1:
        return [f"[FILLER] detected at {mmss(start_time)}"]
    return []

out_path = wav_path.with_suffix(".txt")
with out_path.open("w") as fp:
    audio_fillers = []
    
    for segment in segments:
        # Extract audio segment
        start_sample = int(segment.start * sr)
        end_sample = int(segment.end * sr)
        audio_seg = audio[start_sample:end_sample]
        
        # Check for audio-based fillers
        audio_fillers.extend(detect_audio_fillers(audio_seg, sr, segment.start))
        
        # Process text and highlight known fillers
        text = segment.text.strip()
        for filler in FILLERS:
            text = re.sub(rf'\b{re.escape(filler)}\b', filler.upper(), text, flags=re.IGNORECASE)
        
        fp.write(f"{mmss(segment.start)} - {mmss(segment.end)}: {text}\n")
    
    # Add detected audio fillers
    for filler in audio_fillers:
        fp.write(f"{filler}\n")