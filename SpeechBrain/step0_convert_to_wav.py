import subprocess, os
import sys

def ffmpeg_to_wav(src, dst, sr=16000):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-ac", "1", "-ar", str(sr), dst],
        check=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_wav.py input_video.[mp4|mov] [output_audio.wav]")
        sys.exit(1)
    in_path = sys.argv[1]
    if len(sys.argv) > 2:
        wav_path = sys.argv[2]
    else:
        wav_path = os.path.splitext(in_path)[0] + ".wav"
    ffmpeg_to_wav(in_path, wav_path)