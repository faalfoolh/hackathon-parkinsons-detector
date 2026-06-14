"""
Balanced Audio Cleaning - Realistic preprocessing
Moderate noise reduction while preserving voice characteristics
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import subprocess
import tempfile
import os
from scipy import signal

def convert_to_wav(m4a_path):
    """Convert m4a to high-quality WAV"""
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav.close()

    cmd = [
        'ffmpeg', '-i', str(m4a_path),
        '-ar', '44100',  # CD-quality sample rate
        '-ac', '1',      # Mono
        '-acodec', 'pcm_s16le',  # Uncompressed PCM
        '-y', temp_wav.name
    ]

    subprocess.run(cmd, capture_output=True, check=True)
    return temp_wav.name

def balanced_noise_reduction(audio, sr):
    """
    Moderate noise reduction using spectral subtraction
    """
    # Compute short-time Fourier transform
    hop_length = 512
    n_fft = 2048
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Estimate noise from quieter frames (bottom 10%)
    frame_power = np.sum(magnitude**2, axis=0)
    noise_threshold = np.percentile(frame_power, 10)
    noise_frames = magnitude[:, frame_power < noise_threshold]

    if noise_frames.shape[1] > 0:
        noise_profile = np.mean(noise_frames, axis=1, keepdims=True)
    else:
        noise_profile = np.percentile(magnitude, 10, axis=1, keepdims=True)

    # Spectral subtraction with over-subtraction factor
    alpha = 1.5  # Moderate over-subtraction
    magnitude_clean = magnitude - alpha * noise_profile
    magnitude_clean = np.maximum(magnitude_clean, 0.1 * magnitude)  # Floor to prevent artifacts

    # Reconstruct audio
    stft_clean = magnitude_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=hop_length, n_fft=n_fft)

    return audio_clean

def highpass_filter(audio, sr, cutoff=60):
    """
    Gentle highpass to remove very low rumble (not voice)
    60 Hz removes electrical hum but preserves all voice
    """
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist

    # 2nd order Butterworth (gentle slope)
    sos = signal.butter(2, normalized_cutoff, btype='high', output='sos')
    filtered = signal.sosfilt(sos, audio)

    return filtered

def normalize_audio(audio, target_db=-20):
    """
    Normalize audio to target dB level
    -20 dB is moderate, not too loud
    """
    # Convert to dB
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        current_db = 20 * np.log10(rms)
        gain_db = target_db - current_db
        gain = 10 ** (gain_db / 20)
        audio = audio * gain

    # Prevent clipping
    peak = np.max(np.abs(audio))
    if peak > 0.95:
        audio = audio * (0.95 / peak)

    return audio

def clean_audio_balanced(input_path, output_path):
    """
    Balanced cleaning pipeline - realistic noise reduction
    """
    print(f"  Processing: {input_path.name}")

    # Convert to WAV if needed
    if str(input_path).endswith('.m4a'):
        wav_path = convert_to_wav(input_path)
    else:
        wav_path = str(input_path)

    try:
        # Load audio at high sample rate to preserve harmonics
        audio, sr = librosa.load(wav_path, sr=44100, mono=True)

        # Step 1: Remove very low frequency rumble (< 60 Hz)
        # This doesn't affect voice (voice starts at ~80 Hz)
        audio = highpass_filter(audio, sr, cutoff=60)

        # Step 2: Moderate noise reduction
        # Reduces background noise but preserves voice harmonics
        audio = balanced_noise_reduction(audio, sr)

        # Step 3: Normalize to consistent level
        audio = normalize_audio(audio, target_db=-20)

        # Save cleaned audio
        sf.write(output_path, audio, sr)
        return True

    finally:
        # Cleanup temporary file
        if str(input_path).endswith('.m4a') and os.path.exists(wav_path):
            os.remove(wav_path)

def process_directory(input_dir, output_dir):
    """Process all audio files with balanced cleaning"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    audio_files = list(input_dir.glob("*.m4a"))

    print("=" * 70)
    print("BALANCED AUDIO CLEANING")
    print("=" * 70)
    print(f"Found {len(audio_files)} audio files")
    print("\nProcessing steps:")
    print("  1. Convert to uncompressed WAV (44.1kHz)")
    print("  2. Remove low-frequency rumble (< 60 Hz)")
    print("  3. Moderate spectral noise reduction")
    print("  4. Normalize volume to -20 dB")
    print("\nPreserving:")
    print("  ✓ All voice harmonics (60 Hz - 22 kHz)")
    print("  ✓ Natural voice characteristics")
    print("  ✓ Pitch and timbre information")
    print()

    success_count = 0
    for audio_file in audio_files:
        output_file = output_dir / f"{audio_file.stem}_balanced.wav"
        try:
            clean_audio_balanced(audio_file, output_file)
            success_count += 1
        except Exception as e:
            print(f"    ERROR processing {audio_file.name}: {e}")
            continue

    print(f"\n✓ Successfully processed {success_count}/{len(audio_files)} files")
    print(f"✓ Cleaned files saved to: {output_dir}")
    print("\nNext step: Extract features from cleaned audio")

if __name__ == "__main__":
    INPUT_DIR = "data/audio"
    OUTPUT_DIR = "data/audio_cleaned_balanced"

    process_directory(INPUT_DIR, OUTPUT_DIR)
