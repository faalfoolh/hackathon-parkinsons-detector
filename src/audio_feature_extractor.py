"""
Audio Feature Extractor for Parkinson's Disease Detection
Extracts voice biomarker features from audio files
"""

import parselmouth
from parselmouth.praat import call
import numpy as np
import pandas as pd
import os
from pathlib import Path
import subprocess
import tempfile

def convert_m4a_to_wav(m4a_path):
    """
    Convert m4a audio file to WAV format using ffmpeg

    Args:
        m4a_path: Path to the m4a file

    Returns:
        Path to the temporary WAV file
    """
    # Create temporary WAV file
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav.close()

    # Convert using ffmpeg
    cmd = [
        'ffmpeg',
        '-i', str(m4a_path),
        '-ar', '16000',  # Sample rate 16kHz (good for voice)
        '-ac', '1',       # Mono
        '-y',             # Overwrite output
        temp_wav.name
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return temp_wav.name
    except subprocess.CalledProcessError as e:
        print(f"    Error converting {m4a_path}: {e}")
        return None

def extract_features_from_audio(audio_path):
    """
    Extract voice features from an audio file using Praat/Parselmouth

    Returns a dictionary with all the voice biomarker features
    """
    # Convert m4a to wav if needed
    wav_path = audio_path
    temp_wav_path = None

    if str(audio_path).endswith('.m4a'):
        temp_wav_path = convert_m4a_to_wav(audio_path)
        if temp_wav_path is None:
            raise ValueError(f"Failed to convert {audio_path}")
        wav_path = temp_wav_path

    try:
        # Load the audio file
        sound = parselmouth.Sound(wav_path)

        # Create pitch object for fundamental frequency analysis
        pitch = call(sound, "To Pitch", 0.0, 75, 600)  # time_step, pitch_floor, pitch_ceiling

        # Create harmonicity object for HNR
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)

        # Create point process for jitter/shimmer
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)

        features = {}

        # === FUNDAMENTAL FREQUENCY FEATURES ===
        # Extract pitch statistics
        features['MDVP:Fo(Hz)'] = call(pitch, "Get mean", 0, 0, "Hertz")  # mean F0
        features['MDVP:Fhi(Hz)'] = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")  # max F0
        features['MDVP:Flo(Hz)'] = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")  # min F0

        # === JITTER FEATURES (pitch variation) ===
        # Local jitter
        # Note: Praat returns jitter as a ratio (e.g., 0.006 = 0.6%), NOT percentage
        # The Parkinsons dataset stores it as a ratio, so don't multiply by 100
        features['MDVP:Jitter(%)'] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features['MDVP:Jitter(Abs)'] = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)

        # RAP - Relative Average Perturbation
        features['MDVP:RAP'] = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)

        # PPQ - Five-point Period Perturbation Quotient
        features['MDVP:PPQ'] = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)

        # DDP - Average absolute difference of differences between jitter cycles
        features['Jitter:DDP'] = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

        # === SHIMMER FEATURES (amplitude variation) ===
        # Local shimmer
        features['MDVP:Shimmer'] = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # Shimmer in dB
        features['MDVP:Shimmer(dB)'] = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # APQ3 - Three-point Amplitude Perturbation Quotient
        features['Shimmer:APQ3'] = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # APQ5 - Five-point Amplitude Perturbation Quotient
        features['Shimmer:APQ5'] = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # APQ - 11-point Amplitude Perturbation Quotient
        features['MDVP:APQ'] = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # DDA - Average absolute difference between consecutive differences
        features['Shimmer:DDA'] = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # === NOISE FEATURES ===
        # NHR - Noise-to-Harmonics Ratio
        # Calculate using mean harmonicity
        mean_hnr = call(harmonicity, "Get mean", 0, 0)
        features['HNR'] = mean_hnr

        # NHR is approximately the inverse of HNR in linear scale
        # Convert HNR (dB) to linear, take inverse, normalize
        if mean_hnr > 0:
            features['NHR'] = 10 ** (-mean_hnr / 20)  # Convert dB to linear ratio
        else:
            features['NHR'] = 1.0  # Default if HNR is 0 or negative

        return features

    finally:
        # Cleanup temporary WAV file if created
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)


def process_audio_directory(audio_dir, output_csv):
    """
    Process all audio files in a directory and save features to CSV

    Args:
        audio_dir: Directory containing .m4a audio files
        output_csv: Path to save the output CSV file
    """
    audio_dir = Path(audio_dir)
    audio_files = sorted(list(audio_dir.glob("*.m4a")))

    if not audio_files:
        print(f"No .m4a files found in {audio_dir}")
        return

    print(f"Found {len(audio_files)} audio files")
    print("Extracting features...")

    all_features = []

    for audio_file in audio_files:
        print(f"  Processing: {audio_file.name}")
        try:
            features = extract_features_from_audio(str(audio_file))
            features['name'] = audio_file.name
            all_features.append(features)
        except Exception as e:
            print(f"    ERROR processing {audio_file.name}: {e}")
            continue

    if not all_features:
        print("No features extracted successfully!")
        return

    # Create DataFrame
    df = pd.DataFrame(all_features)

    # Reorder columns to match expected format
    column_order = [
        'name',
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
        'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
        'NHR', 'HNR'
    ]

    df = df[column_order]

    # Add placeholder columns for features that will be imputed later
    # (status, RPDE, DFA, spread1, spread2, D2, PPE)
    df['status'] = np.nan  # Will be set to 0 by extract_features.py
    df['RPDE'] = np.nan
    df['DFA'] = ''
    df['spread1'] = ''
    df['spread2'] = ''
    df['D2'] = ''
    df['PPE'] = ''

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Successfully extracted features from {len(all_features)} files")
    print(f"✓ Saved to: {output_csv}")
    print(f"\nNext step: Run 'python src/extract_features.py' to generate ground truth file")


if __name__ == "__main__":
    # Configuration
    AUDIO_DIR = "data/audio"
    OUTPUT_CSV = "data/voice_features_database.csv"

    print("=" * 60)
    print("Audio Feature Extraction for Parkinson's Detection")
    print("=" * 60)

    process_audio_directory(AUDIO_DIR, OUTPUT_CSV)
