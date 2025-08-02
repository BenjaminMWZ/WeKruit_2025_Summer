#!/usr/bin/env python
"""
Transcribe a *single-speaker* WAV and tag filler words.

Usage
-----
python transcribe_fillers.py  interviewee.wav

Outputs
-------
<base>.txt          – full transcript (“At MM:SS Speaker: …”)
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

import pathlib, itertools, soundfile as sf, librosa
from transformers import (pipeline, AutoProcessor,
                          AutoModelForSpeechSeq2Seq)

# small, English-only Whisper checkpoint (~ 244 MB)
WH_ID = "openai/whisper-small.en"     #  [oai_citation:0‡Hugging Face](https://huggingface.co/openai/whisper-small.en/blame/refs%2Fpr%2F9/README.md?utm_source=chatgpt.com)

proc  = AutoProcessor.from_pretrained(WH_ID)
if proc.tokenizer.pad_token_id is None:            # avoid pipeline crash
    proc.tokenizer.pad_token_id = proc.tokenizer.eos_token_id
model = AutoModelForSpeechSeq2Seq.from_pretrained(WH_ID)

asr = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=proc.tokenizer,
    feature_extractor=proc.feature_extractor,
    return_timestamps="word",
    chunk_length_s=30,
    device="cpu",
)

FILLERS = {"um", "uh", "erm", "hmm", "ah", "eh", "like", "you know"}

def mmss(t): m,s=divmod(int(t),60); return f"{m:02d}:{s:02d}"

if len(sys.argv) < 2:
    sys.exit("Usage:  python transcribe_fillers.py  speaker.wav")

wav_path = pathlib.Path(sys.argv[1]).expanduser()
wav, sr  = sf.read(wav_path)
if sr != 16_000:
    wav = librosa.resample(wav, sr, 16_000); sr = 16_000
tmp = wav_path.with_suffix(".16k.wav")
sf.write(tmp, wav, 16_000)

out = asr(str(tmp), batch_size=8)         
words = out["chunks"]

def groups(word_list):
    """Consecutive words → (start, text)"""
    for _, g in itertools.groupby(word_list,
                                  lambda w: int(w["timestamp"][0]//5)):  # 5-s bins
        g = list(g)
        yield g[0]["timestamp"][0], " ".join(w["text"] for w in g)

txt_path = wav_path.with_suffix(".txt")
with txt_path.open("w") as fp:
    for start, text in groups(words):
        # simple filler highlight
        for f in FILLERS:
            text = text.replace(f, f.upper())
        fp.write(f'At {mmss(start)}, Speaker: "{text}"\n')

print("✓ transcript saved →", txt_path.name)