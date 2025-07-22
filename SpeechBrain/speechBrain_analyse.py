#!/usr/bin/env python
"""
End-to-end interview-audio analysis (no heavy SpeechScorer):

1. Fast 2-speaker diarization   (WebRTC VAD + SpeechBrain ECAPA + 2-cluster HAC)
2. Whisper-tiny ASR → filler-word timestamps
3. Heuristic audio-feature scores → fluency % & intelligibility %
4. Heuristic MFCC/pitch features → accent top-3

Outputs speech_metrics.json in the current folder.
"""

# ────────────────────── env tweaks ──────────────────────────
import os
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"   # avoid old TBB crash

# ────────────────────── imports ─────────────────────────────
import sys, json, tempfile, subprocess, numpy as np, torch, soundfile as sf
from tqdm import tqdm
import webrtcvad, librosa
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference import EncoderClassifier          # for ECAPA embeds
import whisper                                               # tiny model

# ────────── constants & simple helpers ─────────────────────
FILLERS = {"um", "uh", "erm", "hmm", "like", "know", "so",
           "actually", "basically", "right", "i mean"}

def ffmpeg_to_wav(src, dst, sr=16000):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-ac", "1", "-ar", str(sr), dst],
        check=True)

def webrtc_vad_segments(wav, sr, frame_ms=30):
    vad = webrtcvad.Vad(2)
    frame_len = int(sr * frame_ms / 1000)
    pcm16 = (wav * 32767).astype(np.int16).tobytes()
    speech, start = [], None
    for i in range(0, len(pcm16), frame_len*2):
        frame = pcm16[i:i+frame_len*2]
        if len(frame) < frame_len*2:
            break
        if vad.is_speech(frame, sr):
            if start is None:
                start = i // 2 / sr
        else:
            if start is not None and (i//2/sr - start) >= 0.3:
                speech.append((start, i//2/sr))
            start = None
    if start is not None:
        speech.append((start, len(wav)/sr))
    return speech

def transcribe_chunks(model, chunks, full_wav, sr):
    """Run Whisper on a list of (start, stop) tuples and concatenate."""
    if not chunks:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio = np.concatenate([full_wav[int(s*sr):int(e*sr)] for s, e in chunks])
        sf.write(tmp.name, audio, sr)
        res = model.transcribe(tmp.name, fp16=False)
    return res["text"].strip()

def ts(sec):                 # nice MM:SS string
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"

# ────────── heuristic quality metrics ───────────────────────
def analyze_speech_quality(audio, sr):
    rms = np.sqrt(np.mean(audio ** 2))
    vad_ratio = np.sum(audio ** 2 > 0.01 * np.max(audio ** 2)) / len(audio)

    stft_mag = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    centroid = np.mean(librosa.feature.spectral_centroid(S=stft_mag))
    rolloff  = np.mean(librosa.feature.spectral_rolloff(S=stft_mag))
    balance  = centroid / rolloff if rolloff else 0

    energy_var = np.var(librosa.feature.rms(y=audio))

    flu  = min(100, int(60 + vad_ratio*30 + (1-energy_var)*20))
    intel= min(100, int(50 + balance*25 + vad_ratio*25))
    return flu, intel

def analyze_accent_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    pitch = librosa.yin(audio, fmin=50, fmax=400)
    p_std = np.nanstd(pitch)
    sc    = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

    acc = [("Indian", 0.4) if p_std > 50 else ("American", 0.5),
           ("British", 0.3) if sc < 2000 else ("Australian", 0.2),
           ("Canadian", 0.1)]
    return acc[:3]

# ────────── main ────────────────────────────────────────────
if len(sys.argv) < 2:
    sys.exit("Usage: python analyze_interview.py <audio|video file>")

in_path = sys.argv[1]
if not os.path.isfile(in_path):
    sys.exit(f"File not found: {in_path}")

# 1) ensure 16-kHz mono WAV
if in_path.lower().endswith(".mp4"):
    wav_path = os.path.splitext(in_path)[0] + ".wav"
    ffmpeg_to_wav(in_path, wav_path)
else:
    wav_path = in_path

wav, sr = sf.read(wav_path)
if sr != 16000:
    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    sr  = 16000

# 2) fast 2-speaker diarization
speech_chunks = webrtc_vad_segments(wav, sr)
if not speech_chunks:
    sys.exit("No speech detected.")

enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                     run_opts={"device": "cpu"},
                                     savedir="pretrained_models/spk-ecapa")
embs = []
for s, e in speech_chunks:
    seg = torch.from_numpy(wav[int(s*sr):int(e*sr)]).unsqueeze(0)
    with torch.no_grad():
        embs.append(enc.encode_batch(seg).squeeze(0).numpy())
X = np.vstack(embs)

labels = AgglomerativeClustering(n_clusters=2, metric="cosine",
                                 linkage="average").fit_predict(X)
dur   = {}
for (s,e), lbl in zip(speech_chunks, labels):
    dur[lbl] = dur.get(lbl,0)+ (e-s)
inter_lbl = max(dur, key=dur.get)
inter_segments = [(s,e) for (s,e), lbl in zip(speech_chunks, labels)
                  if lbl==inter_lbl]
inter_wav = np.concatenate([wav[int(s*sr):int(e*sr)] for s,e in inter_segments])

interviewee_lbl = inter_lbl  # Define interviewee_lbl as the label with longest duration

with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    sf.write(tmp.name, inter_wav, sr)
    inter_wav_path = tmp.name

# ------------------------------------------------------------------
# TRANSCRIBE BOTH SPEAKERS FOR DEBUGGING
# ------------------------------------------------------------------
wh_model = whisper.load_model("tiny")

interviewer_segments = [(s, e) for (s, e), lbl in zip(speech_chunks, labels)
                        if lbl != interviewee_lbl]

interviewee_txt  = transcribe_chunks(wh_model, inter_segments,  wav, sr)
interviewer_txt  = transcribe_chunks(wh_model, interviewer_segments, wav, sr)

with open("speaker_dialogue.txt", "w") as fp:
    fp.write(f'Interviewee: "{interviewee_txt}"\n')
    fp.write(f'Interviewer: "{interviewer_txt}"\n')

print("✓ dialogue file saved → speaker_dialogue.txt")
# ------------------------------------------------------------------

# 3) Whisper-tiny ASR → filler timestamps
wh = whisper.load_model("tiny")
asr  = wh.transcribe(inter_wav_path, word_timestamps=True, fp16=False)

fillers = []
for seg in asr["segments"]:
    for w in seg["words"]:
        word = w["word"].lower().strip(" ,.!?")
        if word in FILLERS:
            fillers.append(f"{w['word'].strip()} detected at {ts(w['start'])}")

# 4) heuristic fluency + intelligibility
flu_pct, intel_pct = analyze_speech_quality(inter_wav, sr)

# 5) heuristic accent
accent_top3 = analyze_accent_features(inter_wav, sr)

# 6) save JSON
out = {
    "filler_words"      : fillers,
    "fluency_percent"   : flu_pct,
    "intelligibility_percent": intel_pct,
    "accent_top3"       : accent_top3,
    "metadata"          : {
        "audio_seconds": round(len(inter_wav)/sr,1),
        "sample_rate"  : sr
    }
}
with open("speech_metrics.json", "w") as f:
    json.dump(out, f, indent=2)
print("✓ analysis finished → speech_metrics.json")