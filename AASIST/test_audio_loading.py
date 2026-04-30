"""
Test script to verify audio files can be loaded from ASVspoof 2021 dataset.

Usage:
    python test_audio_loading.py \
        --protocol_csv ./asvspoof2021_eval_whisper_merged_with_labels.csv \
        --audio_dir /path/to/audio/files \
        --num_test 10
"""

import argparse
import csv
from pathlib import Path
import soundfile as sf
import numpy as np


def test_audio_loading(protocol_csv, audio_dir, num_test=10, subset='eval'):
    """Test loading audio files."""

    print("="*60)
    print("Audio Loading Diagnostic Test")
    print("="*60)

    audio_dir = Path(audio_dir)
    print(f"\nAudio directory: {audio_dir}")
    print(f"Directory exists: {audio_dir.exists()}")

    if audio_dir.exists():
        print(f"Directory is readable: {audio_dir.is_dir()}")
        # Try to list first few files
        try:
            files = list(audio_dir.glob('*.flac'))[:5]
            if not files:
                files = list(audio_dir.glob('*.wav'))[:5]
            print(f"Sample files found: {len(files)}")
            for f in files:
                print(f"  - {f.name}")
        except Exception as e:
            print(f"Error listing files: {e}")

    print("\n" + "-"*60)
    print("Loading utterance IDs from CSV...")
    print("-"*60)

    # Load utterance IDs from CSV
    utterance_ids = []
    with open(protocol_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if subset == 'all' or row['subset'] == subset:
                utterance_ids.append(row['utterance_id'])
                if len(utterance_ids) >= num_test:
                    break

    print(f"Testing {len(utterance_ids)} utterances...")

    # Test loading each file
    success_count = 0
    fail_count = 0

    for i, utt_id in enumerate(utterance_ids, 1):
        print(f"\n[{i}/{len(utterance_ids)}] Testing: {utt_id}")

        # Try to find the file
        audio_path = None
        for ext in ['.flac', '.wav']:
            # Try direct path
            path = audio_dir / f"{utt_id}{ext}"
            if path.exists():
                audio_path = path
                print(f"  ✓ Found: {path}")
                break

            # Try with flac subdirectory
            path = audio_dir / "flac" / f"{utt_id}{ext}"
            if path.exists():
                audio_path = path
                print(f"  ✓ Found: {path}")
                break

        if audio_path is None:
            print(f"  ✗ File not found!")
            print(f"    Tried:")
            print(f"      - {audio_dir / f'{utt_id}.flac'}")
            print(f"      - {audio_dir / f'{utt_id}.wav'}")
            print(f"      - {audio_dir / 'flac' / f'{utt_id}.flac'}")
            print(f"      - {audio_dir / 'flac' / f'{utt_id}.wav'}")
            fail_count += 1
            continue

        # Try to load the audio
        try:
            data, samplerate = sf.read(str(audio_path))
            print(f"  ✓ Loaded successfully!")
            print(f"    Sample rate: {samplerate} Hz")
            print(f"    Duration: {len(data)/samplerate:.2f} seconds")
            print(f"    Shape: {data.shape}")
            print(f"    Data type: {data.dtype}")
            print(f"    Value range: [{np.min(data):.4f}, {np.max(data):.4f}]")
            success_count += 1
        except Exception as e:
            print(f"  ✗ Error loading file: {e}")
            print(f"    Error type: {type(e).__name__}")
            fail_count += 1

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Total tested: {len(utterance_ids)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Success rate: {success_count/len(utterance_ids)*100:.1f}%")

    if fail_count > 0:
        print("\n⚠ Warning: Some files failed to load!")
        print("Possible issues:")
        print("  1. Incorrect audio directory path")
        print("  2. Missing audio files")
        print("  3. Corrupted audio files")
        print("  4. Incorrect file naming convention")
        print("  5. Insufficient permissions")
    else:
        print("\n✓ All files loaded successfully!")
        print("You should be able to run evaluate_2021.py now.")

    return success_count, fail_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test audio file loading for ASVspoof 2021'
    )

    parser.add_argument('--protocol_csv', type=str, required=True,
                        help='Path to protocol CSV file')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--num_test', type=int, default=10,
                        help='Number of files to test (default: 10)')
    parser.add_argument('--subset', type=str, default='eval',
                        choices=['eval', 'progress', 'all'],
                        help='Which subset to test (default: eval)')

    args = parser.parse_args()

    test_audio_loading(
        args.protocol_csv,
        args.audio_dir,
        args.num_test,
        args.subset
    )
