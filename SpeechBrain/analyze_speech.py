#!/usr/bin/env python
"""
Analyze *interviewee* audio only:

• Whisper-tiny → filler-word timestamps
• Heuristic audio features → fluency % & intelligibility %
• MFCC / pitch → accent top-3
Outputs speech_metrics.json
"""

import os, sys, json, numpy as np, soundfile as sf, librosa, whisper

os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

FILLERS = {"um","uh","erm","hmm","like","know","so",
           "actually","basically","right","i mean"}

def ts(sec):                       # MM:SS
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"

def analyze_quality(audio, sr):
    vad = np.sum(audio**2 > 0.01*np.max(audio**2)) / len(audio)
    stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    bal  = np.mean(librosa.feature.spectral_centroid(S=stft)) / \
           np.mean(librosa.feature.spectral_rolloff(S=stft))
    energy_var = np.var(librosa.feature.rms(y=audio))
    flu  = min(100, int(60 + vad*30 + (1-energy_var)*20))
    intel= min(100, int(50 + bal*25 + vad*25))
    return flu, intel

def analyze_accent(audio, sr):
    p_std = np.nanstd(librosa.yin(audio, fmin=50, fmax=400))
    sc    = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    lst   = [("Indian",0.4) if p_std>50 else ("American",0.5),
             ("British",0.3) if sc<2000 else ("Australian",0.2),
             ("Canadian",0.1)]
    return lst[:3]

# ---- main ---------------------------------------------------------------
if len(sys.argv) < 2:
    sys.exit("Usage: python analyze_speech.py  <interviewee.wav>")

wav_path = sys.argv[1]
wav, sr  = sf.read(wav_path)
if sr != 16000:
    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000); sr=16000

# 1) Whisper filler-word timestamps
wh = whisper.load_model("tiny", device="cpu")
tmp_json = wh.transcribe(wav_path, word_timestamps=True, fp16=False)

fillers = []
for seg in tmp_json["segments"]:
    for w in seg["words"]:
        wd = w["word"].lower().strip(" ,.!?")
        if wd in FILLERS:
            fillers.append(f"{w['word'].strip()} detected at {ts(w['start'])}")

# 2) heuristic scores
flu, intel = analyze_quality(wav, sr)

# 3) accent
accent = analyze_accent(wav, sr)

out = {
    "filler_words": fillers,
    "fluency_percent": flu,
    "intelligibility_percent": intel,
    "accent_top3": accent,
    "metadata": {
        "audio_seconds": round(len(wav)/sr,1),
        "sample_rate": sr
    }
}
with open("speech_metrics.json", "w") as f:
    json.dump(out, f, indent=2)

print("✓ analysis finished → speech_metrics.json")