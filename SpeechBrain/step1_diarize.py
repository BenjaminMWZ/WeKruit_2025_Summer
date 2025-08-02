#!/usr/bin/env python
"""
Split a mixed 2-speaker interview into separate WAV files.

Usage
-----
python diarize_split.py  interview.wav    # or .mp4

Outputs
-------
<base>_interviewee.wav
<base>_interviewer.wav
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

import pathlib, subprocess, librosa, soundfile as sf, numpy as np
from pyannote.audio import Pipeline         #  requires HF TOKEN

HF_TOKEN   = os.getenv("HF_TOKEN") or sys.exit("export HF_TOKEN=<token>")
DIAR_MODEL = "pyannote/speaker-diarization-3.1"


# ---------- helpers ----------------------------------------------------
def ffmpeg_to_wav(src, dst, sr=16_000):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                    "-ac", "1", "-ar", str(sr), dst], check=True)

def save_wav(x, sr, path): sf.write(path, x, sr)


# ---------- main -------------------------------------------------------
if len(sys.argv) < 2:
    sys.exit("Usage:  python diarize_split.py  interview.wav")

src   = pathlib.Path(sys.argv[1]).expanduser()
base  = src.with_suffix("")                                  # no ext

# 1) make 16-kHz mono WAV ---------------------------------------------
wav_path = base.with_suffix(".wav")
if src.suffix.lower() == ".mp4":
    ffmpeg_to_wav(src, wav_path)
else:
    wav_path = src

wav, sr = sf.read(wav_path)
if sr != 16_000:
    wav = librosa.resample(wav, sr, 16_000); sr = 16_000

# 2)  diarize -----------------------------------------------------------
dia   = Pipeline.from_pretrained(DIAR_MODEL, use_auth_token=HF_TOKEN)
annot = dia(str(wav_path))

# choose longest speaker as “interviewee”
dur = {}
for seg, _, spk in annot.itertracks(yield_label=True):
    dur[spk] = dur.get(spk, 0.) + (seg.end - seg.start)
assert dur, "No speech found."
interviewee = max(dur, key=dur.get)

# gather segments
tracks = {sp: [] for sp in annot.labels()}
for seg, _, spk in annot.itertracks(yield_label=True):
    tracks[spk].append(seg)

def cat(seglist):
    return np.concatenate([wav[int(s.start*sr):int(s.end*sr)] for s in seglist])

ivi = cat(tracks[interviewee])
other_spks = [sp for sp in tracks if sp != interviewee]
ivr = cat(tracks[other_spks[0]]) if other_spks else np.array([])

ivi_path = base.parent / f"{base.stem}_interviewee.wav"
save_wav(ivi, sr, ivi_path)

if ivr.size:
    ivr_path = base.parent / f"{base.stem}_interviewer.wav"
    save_wav(ivr, sr, ivr_path)
    print("✓ diarization: 2 tracks saved ->", ivi_path.name, ivr_path.name)
else:
    print("✓ diarization: only one speaker detected (interviewee)")