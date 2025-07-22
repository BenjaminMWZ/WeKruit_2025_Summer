#!/usr/bin/env python
"""
Fast 2-speaker diarization + filler-aware transcription.

Outputs, next to the input file:
    <base>_interviewee.wav
    <base>_interviewer.wav
    <base>_dialogue.txt
"""

# ───── boot block (must stay at the very top of the script) ────────────
import os, sys, types, importlib.machinery
os.environ["TRANSFORMERS_NO_TF"] = "1"          # never load real TensorFlow
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"   # old-TBB fix

class _TFStub(types.ModuleType):
    """Lazy TensorFlow stub which satisfies Torch + Transformers."""
    def __init__(self, name):
        super().__init__(name)
        self.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        self.__file__ = "<stub>"                # lets inspect.getsourcefile() work
        # --- minimal types expected by transformers.tf_utils -------------
        class _Dummy:                                   # pylint: disable=too-few-public-methods
            pass
        self.Tensor   = _Dummy                          # ← NEW
        self.Variable = _Dummy                          # ← NEW

    def __getattr__(self, item):                        # auto-create sub-modules
        sub = _TFStub(f"{self.__name__}.{item}")
        setattr(self, item, sub)
        sys.modules[sub.__name__] = sub
        return sub

tf_stub = _TFStub("tensorflow")
sys.modules["tensorflow"] = tf_stub
# ────────────────────────────────────────────────────────────────────────

import pathlib, subprocess, itertools, tempfile
import numpy as np, soundfile as sf, librosa, torch, whisper

from pyannote.audio import Pipeline                         # diarization
from transformers import (pipeline, AutoProcessor,
                          AutoModelForSpeechSeq2Seq)

# ───────── config ──────────────────────────────────────────
HF_TOKEN  = os.getenv("HF_TOKEN") or sys.exit("export HF_TOKEN=<token>")
DIAR_MODEL= "pyannote/speaker-diarization-3.1"
# --- CrisperWhisper ----------------------------------------------------
CRISPER_ID = "nyrahealth/CrisperWhisper"

proc   = AutoProcessor.from_pretrained(CRISPER_ID)        # WhisperProcessor
model  = AutoModelForSpeechSeq2Seq.from_pretrained(CRISPER_ID)

asr = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=proc.tokenizer,              # <- give the tokenizer only
    feature_extractor=proc.feature_extractor,
    return_timestamps="word",
    chunk_length_s=30,
    device="cpu",
)
WH_MODEL  = whisper.load_model("tiny", device="cpu") # only for tiny helper

# ───────── helpers ─────────────────────────────────────────
def ffmpeg_to_wav(src, dst, sr=16000):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-ac", "1", "-ar", str(sr), dst], check=True)

def save_wav(arr, sr, path):  sf.write(path, arr, sr)

def mmss(t):                  # 75.3 → 01:15
    m, s = divmod(int(t), 60); return f"{m:02d}:{s:02d}"

def concat_segments(track, wav, sr):
    """Concatenate numpy samples of a list[Segment]."""
    if not track: return np.array([])
    parts = [wav[int(s.start*sr):int(s.end*sr)] for s in track]
    return np.concatenate(parts)

def group_consecutive(words, speaker_of):
    """Yield (spk, start_time, text) for consecutive words of same speaker."""
    keyfunc = lambda w: speaker_of(w["start"])
    for _, grp in itertools.groupby(words, keyfunc):
        grp = list(grp)
        spk = speaker_of(grp[0]["start"])
        yield spk, grp[0]["start"], " ".join(w["word"] for w in grp).strip()

# ───────── main ────────────────────────────────────────────
if len(sys.argv) < 2:
    sys.exit("Usage: python diarize_transcribe.py <audio|video file>")

SRC   = pathlib.Path(sys.argv[1]).expanduser()
BASE  = SRC.with_suffix("")                                   # strip ext

# 1) ensure 16-kHz mono WAV --------------------------------
wav_path = BASE.with_suffix(".wav")
if SRC.suffix.lower() == ".mp4":
    ffmpeg_to_wav(str(SRC), str(wav_path))
else:
    wav_path = SRC

wav, sr = sf.read(wav_path)
if sr != 16000:
    wav = librosa.resample(wav, sr, 16000); sr = 16000

# 2) diarization -------------------------------------------
dia   = Pipeline.from_pretrained(DIAR_MODEL, use_auth_token=HF_TOKEN)
annot = dia(str(wav_path))

# durations per speaker
dur = {}
for seg, _, spk in annot.itertracks(yield_label=True):
    dur[spk] = dur.get(spk, 0.0) + (seg.end - seg.start)

if len(dur) == 0:
    sys.exit("No speech found by diarizer.")
elif len(dur) == 1:
    # edge-case: only one speaker => treat as interviewee
    interviewee = list(dur.keys())[0]
else:
    interviewee = max(dur, key=dur.get)  # longest talker

roles = {sp: ("Interviewee" if sp == interviewee else "Interviewer")
         for sp in annot.labels()}

# gather segments
tracks = {sp: [] for sp in annot.labels()}
for seg, _, spk in annot.itertracks(yield_label=True):
    tracks[spk].append(seg)

ivi = concat_segments(tracks[interviewee], wav, sr)
others = [sp for sp in tracks if sp != interviewee]
ivr = concat_segments(tracks[others[0]], wav, sr) if others else np.array([])

ivi_path = BASE.parent / f"{BASE.stem}_interviewee.wav"
save_wav(ivi, sr, ivi_path)

if ivr.size:
    ivr_path = BASE.parent / f"{BASE.stem}_interviewer.wav"
    save_wav(ivr, sr, ivr_path)
    print("✓ diarization: 2 tracks saved ->",
          ivi_path.name, ivr_path.name)
else:
    print("✓ diarization: only one speaker (interviewee) detected")

# 3) CrisperWhisper transcription --------------------------
proc  = AutoProcessor.from_pretrained(CRISPER_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(CRISPER_ID)
asr   = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=proc,
    feature_extractor=proc,
    return_timestamps="word",
    chunk_length_s=30,
    device="cpu"
)

result = asr({"path": str(wav_path)}, batch_size=8)
words  = result["chunks"]                         # list of dicts

# map timestamp → speaker
def speaker_of(t):
    for seg, _, spk in annot.itertracks(yield_label=True):
        if seg.start <= t < seg.end:
            return roles[spk]
    return "Unknown"

# 4) dialogue file -----------------------------------------
dlg_path = BASE.parent / f"{BASE.stem}_dialogue.txt"
with dlg_path.open("w") as fp:
    word_objs = [{"start": w["timestamp"][0], "word": w["text"]} for w in words]
    for role, start, text in group_consecutive(word_objs, speaker_of):
        fp.write(f'At {mmss(start)}, {role} said: "{text}"\n')

print("✓ dialogue saved →", dlg_path.name)