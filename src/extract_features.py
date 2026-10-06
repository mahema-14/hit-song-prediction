"""Raw-audio pipeline (as in the CS229 paper): 15 s repeated chorus -> 518 features.

11 librosa features (74 raw dims) x 7 statistics (min, mean, median, max, std,
skew, kurtosis) = 518 numbers per song.

Usage:
    python src/extract_features.py --audio_dir audio/ --labels labels.csv --out data/chorus_features.csv
labels.csv: columns  filename,target   (target: 1 hit, 0 non-hit)
"""
import argparse, os, tempfile
import numpy as np, pandas as pd, librosa
from scipy.stats import skew, kurtosis

CLIP = 15
SR = 22050


def find_chorus(path):
    """Return 15 s chorus audio via PyChorus; fall back to the middle 15 s."""
    y, sr = librosa.load(path, sr=SR)
    try:
        from pychorus import find_and_output_chorus
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "chorus.wav")
            start = find_and_output_chorus(path, out, CLIP)
            if start is not None and os.path.exists(out):
                c, _ = librosa.load(out, sr=SR)
                return c, True
    except Exception:
        pass
    n = CLIP * sr
    mid = max(0, len(y) // 2 - n // 2)
    return y[mid: mid + n], False


def feature_matrices(y, sr=SR):
    return {
        "chroma_stft": librosa.feature.chroma_stft(y=y, sr=sr),
        "chroma_cqt": librosa.feature.chroma_cqt(y=y, sr=sr),
        "chroma_cens": librosa.feature.chroma_cens(y=y, sr=sr),
        "mfcc": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20),
        "rms": librosa.feature.rms(y=y),
        "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=sr),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr),
        "spectral_contrast": librosa.feature.spectral_contrast(y=y, sr=sr),
        "spectral_rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr),
        "tonnetz": librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr),
        "zero_crossing_rate": librosa.feature.zero_crossing_rate(y),
    }


STATS = {"min": np.min, "mean": np.mean, "median": np.median, "max": np.max,
         "std": np.std, "skew": skew, "kurtosis": kurtosis}


def extract(path):
    y, found = find_chorus(path)
    feats = {}
    for name, M in feature_matrices(y).items():
        for i, row in enumerate(M):
            for s, fn in STATS.items():
                feats[f"{s}_{name}.{i}"] = float(fn(row))
    return feats, found


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="data/chorus_features.csv")
    a = ap.parse_args()
    lab = pd.read_csv(a.labels)
    rows = []
    for _, r in lab.iterrows():
        f, found = extract(os.path.join(a.audio_dir, r["filename"]))
        f.update(filename=r["filename"], chorus_found=found, target=r["target"])
        rows.append(f)
        print(r["filename"], "chorus" if found else "fallback(middle 15s)", len(f) - 3, "features")
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("Saved", a.out)
