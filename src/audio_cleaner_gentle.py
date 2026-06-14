"""
Gentle Audio Cleaning - Minimal preprocessing
Only noise reduction and normalization, no aggressive filtering
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import subprocess
import tempfile
import os

def convert_to_wav(m4a_path):
    """Convert m4a to WAV format"""
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav.close()

    cmd = [
        'ffmpeg', '-i', str(m4a_path),
        '-ar', '44100',  # Higher sample rate to preserve harmonics
        '-ac', '1',      # Mono
        '-y', temp_wav.name
    ]

    subprocess.run(cmd, capture_output=True, check=True)
    return temp_wav.name

def gentle_noise_reduction(audio, sr):
    """
    Very gentle noise reduction - only remove extreme noise
    """
    # Compute STFT
    stft = librosa.stft(audio, n_fft=2048)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Very conservative noise floor estimation
    noise_floor = np.percentile(magnitude, 5, axis=1, keepdims=True)

    # Gentle gating - only remove very quiet noise
    threshold = noise_floor * 2  # Less aggressive
    mask = magnitude > threshold
    magnitude_clean = magnitude * (0.1 + 0.9 * mask)  # Soft gating

    # Reconstruct
    stft_clean = magnitude_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean)

    return audio_clean

def normalize_audio(audio):
    """Gentle normalization"""
    # RMS normalization
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        target_rms = 0.05  # Lower target to avoid over-amplification
        audio = audio * (target_rms / rms)

    # Peak normalization
    peak = np.max(np.abs(audio))
    if peak > 0.9:
        audio = audio * (0.9 / peak)

    return audio

def clean_audio_file_gentle(input_path, output_path):
    """
    Minimal cleaning pipeline
    """
    print(f"  Cleaning: {input_path.name}")

    # Convert to WAV
    if str(input_path).endswith('.m4a'):
        wav_path = convert_to_wav(input_path)
    else:
        wav_path = str(input_path)

    try:
        # Load audio at higher sample rate
        audio, sr = librosa.load(wav_path, sr=44100, mono=True)

        # Only gentle processing
        audio = normalize_audio(audio)
        audio = gentle_noise_reduction(audio, sr)
        audio = normalize_audio(audio)

        # Save
        sf.write(output_path, audio, sr)
        return True

    finally:
        if str(input_path).endswith('.m4a') and os.path.exists(wav_path):
            os.remove(wav_path)

def process_directory(input_dir, output_dir):
    """Clean all audio files"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    audio_files = list(input_dir.glob("*.m4a"))

    print("=" * 70)
    print("GENTLE AUDIO CLEANING")
    print("=" * 70)
    print(f"Found {len(audio_files)} files")
    print("\nMinimal processing:")
    print("  1. Volume normalization (gentle)")
    print("  2. Light noise reduction (preserves harmonics)")
    print("  3. Higher sample rate (44.1kHz)")
    print()

    for audio_file in audio_files:
        output_file = output_dir / f"{audio_file.stem}_gentle.wav"
        try:
            clean_audio_file_gentle(audio_file, output_file)
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\n✓ Cleaned files saved to: {output_dir}")

if __name__ == "__main__":
    INPUT_DIR = "data/audio"
    OUTPUT_DIR = "data/audio_cleaned_gentle"

    process_directory(INPUT_DIR, OUTPUT_DIR)
