"""
Audio Cleaning and Preprocessing Script
Cleans audio files to improve voice quality metrics
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
        '-ar', '22050',  # 22kHz sample rate (good for voice)
        '-ac', '1',      # Mono
        '-y', temp_wav.name
    ]

    subprocess.run(cmd, capture_output=True, check=True)
    return temp_wav.name

def reduce_noise_spectral_gating(audio, sr, noise_thresh=0.02):
    """
    Simple spectral noise reduction
    """
    # Compute short-time Fourier transform
    stft = librosa.stft(audio)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Estimate noise floor (from quietest parts)
    noise_floor = np.percentile(magnitude, 20, axis=1, keepdims=True)

    # Apply spectral gating
    mask = magnitude > (noise_floor * (1 + noise_thresh))
    magnitude_clean = magnitude * mask

    # Reconstruct audio
    stft_clean = magnitude_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean)

    return audio_clean

def bandpass_filter(audio, sr, lowcut=80, highcut=500):
    """
    Apply bandpass filter to focus on voice frequencies
    Typical voice: 80-500 Hz for fundamental frequency
    """
    # Design butterworth bandpass filter
    from scipy import signal

    nyquist = sr / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    # Use bandpass filter
    sos = signal.butter(4, [low, high], btype='band', output='sos')
    filtered = signal.sosfilt(sos, audio)

    return filtered

def normalize_audio(audio):
    """Normalize audio to standard level"""
    # RMS normalization
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        target_rms = 0.1  # Target RMS level
        audio = audio * (target_rms / rms)

    # Peak normalization (prevent clipping)
    peak = np.max(np.abs(audio))
    if peak > 0.95:
        audio = audio * (0.95 / peak)

    return audio

def pre_emphasis(audio, coef=0.97):
    """
    Apply pre-emphasis filter to boost higher frequencies
    This can help with voice clarity
    """
    emphasized = np.append(audio[0], audio[1:] - coef * audio[:-1])
    return emphasized

def clean_audio_file(input_path, output_path):
    """
    Complete audio cleaning pipeline
    """
    print(f"  Cleaning: {input_path.name}")

    # Convert to WAV if needed
    if str(input_path).endswith('.m4a'):
        wav_path = convert_to_wav(input_path)
    else:
        wav_path = str(input_path)

    try:
        # Load audio
        audio, sr = librosa.load(wav_path, sr=22050, mono=True)

        # Step 1: Normalize
        audio = normalize_audio(audio)

        # Step 2: Noise reduction
        audio = reduce_noise_spectral_gating(audio, sr, noise_thresh=0.1)

        # Step 3: Bandpass filter (focus on voice frequencies)
        audio = bandpass_filter(audio, sr, lowcut=75, highcut=500)

        # Step 4: Normalize again after filtering
        audio = normalize_audio(audio)

        # Step 5: Pre-emphasis (optional, helps with certain features)
        # audio = pre_emphasis(audio, coef=0.97)

        # Save cleaned audio
        sf.write(output_path, audio, sr)

        return True

    finally:
        # Cleanup temp file
        if str(input_path).endswith('.m4a') and os.path.exists(wav_path):
            os.remove(wav_path)

def process_audio_directory(input_dir, output_dir):
    """
    Clean all audio files in a directory
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    audio_files = list(input_dir.glob("*.m4a"))

    print("=" * 70)
    print("AUDIO CLEANING AND PREPROCESSING")
    print("=" * 70)
    print(f"Found {len(audio_files)} audio files")
    print("\nCleaning pipeline:")
    print("  1. Normalize volume")
    print("  2. Reduce background noise (spectral gating)")
    print("  3. Bandpass filter (75-500 Hz for voice)")
    print("  4. Final normalization")
    print()

    success_count = 0
    for audio_file in audio_files:
        output_file = output_dir / f"{audio_file.stem}_cleaned.wav"
        try:
            clean_audio_file(audio_file, output_file)
            success_count += 1
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\n✓ Successfully cleaned {success_count}/{len(audio_files)} files")
    print(f"✓ Cleaned files saved to: {output_dir}")
    print("\nNext: Run feature extraction on cleaned audio")

if __name__ == "__main__":
    INPUT_DIR = "data/audio"
    OUTPUT_DIR = "data/audio_cleaned"

    process_audio_directory(INPUT_DIR, OUTPUT_DIR)
