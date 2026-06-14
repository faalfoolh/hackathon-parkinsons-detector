"""
More Aggressive Audio Cleaning
Stronger noise reduction while still preserving voice quality
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
        '-ar', '44100',
        '-ac', '1',
        '-acodec', 'pcm_s16le',
        '-y', temp_wav.name
    ]

    subprocess.run(cmd, capture_output=True, check=True)
    return temp_wav.name

def aggressive_noise_reduction(audio, sr):
    """
    More aggressive spectral noise reduction
    """
    hop_length = 512
    n_fft = 2048
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Estimate noise from quieter frames (bottom 15%)
    frame_power = np.sum(magnitude**2, axis=0)
    noise_threshold = np.percentile(frame_power, 15)
    noise_frames = magnitude[:, frame_power < noise_threshold]

    if noise_frames.shape[1] > 0:
        noise_profile = np.median(noise_frames, axis=1, keepdims=True)
    else:
        noise_profile = np.percentile(magnitude, 15, axis=1, keepdims=True)

    # More aggressive spectral subtraction
    alpha = 2.5  # Stronger over-subtraction
    beta = 0.02  # Lower floor (more aggressive)
    magnitude_clean = magnitude - alpha * noise_profile
    magnitude_clean = np.maximum(magnitude_clean, beta * magnitude)

    # Reconstruct
    stft_clean = magnitude_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=hop_length, n_fft=n_fft)

    return audio_clean

def wiener_filter(audio, sr):
    """
    Additional Wiener filtering for noise reduction
    """
    stft = librosa.stft(audio, n_fft=2048)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Estimate signal and noise power
    power = magnitude ** 2
    noise_power = np.percentile(power, 10, axis=1, keepdims=True)
    signal_power = np.maximum(power - noise_power, 0)

    # Wiener gain
    wiener_gain = signal_power / (signal_power + noise_power + 1e-10)

    # Apply gain
    magnitude_filtered = magnitude * wiener_gain

    # Reconstruct
    stft_filtered = magnitude_filtered * np.exp(1j * phase)
    audio_filtered = librosa.istft(stft_filtered)

    return audio_filtered

def dynamic_range_compression(audio, threshold=-20, ratio=3.0):
    """
    Compress dynamic range to make voice more consistent
    Helps with shimmer metrics
    """
    # Convert to dB
    audio_abs = np.abs(audio)
    audio_db = 20 * np.log10(audio_abs + 1e-10)

    # Apply compression above threshold
    mask = audio_db > threshold
    compressed_db = audio_db.copy()
    compressed_db[mask] = threshold + (audio_db[mask] - threshold) / ratio

    # Convert back to linear
    gain = 10 ** ((compressed_db - audio_db) / 20)
    audio_compressed = audio * gain

    return audio_compressed

def enhance_harmonics(audio, sr):
    """
    Enhance harmonic content to improve HNR
    """
    # Harmonic-percussive separation
    harmonic, percussive = librosa.effects.hpss(audio, margin=2.0)

    # Boost harmonics slightly, reduce percussive
    enhanced = harmonic * 1.2 + percussive * 0.3

    return enhanced

def bandpass_voice(audio, sr):
    """
    Gentle bandpass for voice range (80-3000 Hz)
    Removes non-voice frequencies but keeps all harmonics
    """
    nyquist = sr / 2
    low = 80 / nyquist
    high = 3000 / nyquist

    # Gentle 2nd order filter
    sos = signal.butter(2, [low, high], btype='band', output='sos')
    filtered = signal.sosfilt(sos, audio)

    return filtered

def normalize_audio(audio, target_db=-18):
    """Normalize to target level"""
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

def clean_audio_aggressive(input_path, output_path):
    """
    More aggressive cleaning pipeline
    """
    print(f"  Processing: {input_path.name}")

    if str(input_path).endswith('.m4a'):
        wav_path = convert_to_wav(input_path)
    else:
        wav_path = str(input_path)

    try:
        # Load audio
        audio, sr = librosa.load(wav_path, sr=44100, mono=True)

        # Step 1: Initial normalization
        audio = normalize_audio(audio, target_db=-18)

        # Step 2: Bandpass to voice range (removes non-voice frequencies)
        audio = bandpass_voice(audio, sr)

        # Step 3: Aggressive spectral noise reduction
        audio = aggressive_noise_reduction(audio, sr)

        # Step 4: Wiener filtering for additional cleanup
        audio = wiener_filter(audio, sr)

        # Step 5: Enhance harmonic content
        audio = enhance_harmonics(audio, sr)

        # Step 6: Dynamic range compression (reduces shimmer)
        audio = dynamic_range_compression(audio, threshold=-20, ratio=2.5)

        # Step 7: Final normalization
        audio = normalize_audio(audio, target_db=-18)

        # Save
        sf.write(output_path, audio, sr)
        return True

    finally:
        if str(input_path).endswith('.m4a') and os.path.exists(wav_path):
            os.remove(wav_path)

def process_directory(input_dir, output_dir):
    """Process all audio files"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    audio_files = list(input_dir.glob("*.m4a"))

    print("=" * 70)
    print("MORE AGGRESSIVE AUDIO CLEANING")
    print("=" * 70)
    print(f"Found {len(audio_files)} audio files")
    print("\nProcessing pipeline:")
    print("  1. Bandpass filter (80-3000 Hz - voice range)")
    print("  2. Aggressive spectral noise reduction")
    print("  3. Wiener filtering")
    print("  4. Harmonic enhancement (improves HNR)")
    print("  5. Dynamic range compression (reduces shimmer)")
    print("  6. Volume normalization")
    print("\nGoal: Maximize similarity to healthy voice patterns")
    print()

    success_count = 0
    for audio_file in audio_files:
        output_file = output_dir / f"{audio_file.stem}_aggressive.wav"
        try:
            clean_audio_aggressive(audio_file, output_file)
            success_count += 1
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\n✓ Successfully processed {success_count}/{len(audio_files)} files")
    print(f"✓ Cleaned files saved to: {output_dir}")

if __name__ == "__main__":
    INPUT_DIR = "data/audio"
    OUTPUT_DIR = "data/audio_cleaned_aggressive"

    process_directory(INPUT_DIR, OUTPUT_DIR)
