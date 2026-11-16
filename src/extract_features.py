"""Extract the 199 EEG features used in the current ES 678 experiment.

The defaults reproduce the uploaded notebook's current segmentation:
2-second windows (256 samples at 128 Hz), with a 3-second step (384 samples).
The script intentionally does not claim to perform filtering, ICA, or artifact
rejection because those operations are not present in the notebook supplied.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import antropy as ant
import numpy as np
import pandas as pd
import pywt
from scipy.signal import csd, welch
from scipy.integrate import trapezoid
from scipy.stats import kurtosis, skew
from tqdm import tqdm

CHANNELS = ["F3", "F4", "Fz", "P3", "P4", "Pz", "O1", "O2"]
FS = 128
FEATURES_PER_CHANNEL = [
    "mean", "std", "variance", "skewness", "kurtosis",
    "theta_power", "alpha_power", "beta_power", "theta_beta_ratio", "theta_alpha_ratio",
    "higuchi_fd", "spectral_entropy", "sample_entropy",
    "wavelet_cA4_energy", "wavelet_cD4_energy", "wavelet_cD3_energy",
    "wavelet_cD2_energy", "wavelet_cD1_energy", "wavelet_entropy",
    "katz_fd", "petrosian_fd", "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
]


def safe_float(value: float) -> float:
    return float(value) if np.isfinite(value) else np.nan


def calculate_wpli(sig1: np.ndarray, sig2: np.ndarray, fs: int = FS) -> float:
    nperseg = min(fs, len(sig1), len(sig2))
    _, pxy = csd(sig1, sig2, fs=fs, nperseg=nperseg)
    imag_pxy = np.imag(pxy)
    return safe_float(np.abs(np.mean(imag_pxy)) / (np.mean(np.abs(imag_pxy)) + 1e-6))


def wavelet_energies(sig: np.ndarray) -> list[float]:
    coeffs = pywt.wavedec(sig, "db4", level=4)
    return [safe_float(np.sum(np.square(c))) for c in coeffs]


def wavelet_entropy(sig: np.ndarray) -> float:
    energies = np.asarray(wavelet_energies(sig), dtype=float)
    total = np.nansum(energies)
    if not np.isfinite(total) or total <= 0:
        return np.nan
    probabilities = energies / total
    probabilities = probabilities[probabilities > 0]
    return safe_float(-np.sum(probabilities * np.log(probabilities + 1e-6)))


def katz_fd(sig: np.ndarray) -> float:
    length = np.sum(np.abs(np.diff(sig)))
    diameter = np.max(np.abs(sig - sig[0]))
    n = len(sig)
    if length <= 0 or diameter <= 0:
        return np.nan
    denominator = np.log10(diameter / length + 1e-6) + np.log10(n)
    if abs(denominator) < 1e-12:
        return np.nan
    return safe_float(np.log10(n) / denominator)


def petrosian_fd(sig: np.ndarray) -> float:
    diff = np.diff(sig)
    n = len(sig)
    n_delta = np.sum(diff[:-1] * diff[1:] < 0)
    denominator = np.log10(n) + np.log10(n / (n + 0.4 * n_delta))
    if abs(denominator) < 1e-12:
        return np.nan
    return safe_float(np.log10(n) / denominator)


def extract_epoch_features(eeg_data: np.ndarray, fs: int = FS) -> np.ndarray:
    """Input shape: (8 channels, samples). Returns 199 features."""
    values: list[float] = []
    channel_metrics: dict[str, dict[str, float]] = {}

    for index, channel in enumerate(CHANNELS):
        sig = np.asarray(eeg_data[index], dtype=float)
        mean = np.mean(sig)
        std = np.std(sig)
        variance = np.var(sig)
        skewness = skew(sig, bias=True)
        kurt = kurtosis(sig, fisher=True, bias=True)
        values.extend(map(safe_float, [mean, std, variance, skewness, kurt]))

        freqs, psd = welch(sig, fs=fs, nperseg=min(fs, len(sig)))

        def band_power(low: float, high: float) -> float:
            mask = (freqs >= low) & (freqs <= high)
            if mask.sum() < 2:
                return np.nan
            return safe_float(trapezoid(psd[mask], freqs[mask]))

        theta = band_power(4, 8)
        alpha = band_power(8, 13)
        beta = band_power(13, 30)
        tbr = theta / (beta + 1e-6) if np.isfinite(theta) and np.isfinite(beta) else np.nan
        tar = theta / (alpha + 1e-6) if np.isfinite(theta) and np.isfinite(alpha) else np.nan
        channel_metrics[channel] = {"theta": theta, "alpha": alpha, "beta": beta}
        values.extend(map(safe_float, [theta, alpha, beta, tbr, tar]))

        hfd = safe_float(ant.higuchi_fd(sig))
        spec_entropy = safe_float(ant.spectral_entropy(sig, sf=fs, method="welch"))
        sample_entropy = safe_float(ant.sample_entropy(sig))
        channel_metrics[channel].update({"hfd": hfd, "spectral_entropy": spec_entropy})
        values.extend([hfd, spec_entropy, sample_entropy])

        values.extend(wavelet_energies(sig))
        values.append(wavelet_entropy(sig))
        values.extend([katz_fd(sig), petrosian_fd(sig)])

        diff1 = np.diff(sig)
        diff2 = np.diff(diff1)
        activity = variance
        mobility = np.std(diff1) / (std + 1e-6)
        complexity = (np.std(diff2) / (np.std(diff1) + 1e-6)) / (mobility + 1e-6)
        values.extend(map(safe_float, [activity, mobility, complexity]))

    pairs = [("F3", "F4"), ("P3", "P4"), ("Fz", "Pz"), ("O1", "O2")]
    channel_index = {name: i for i, name in enumerate(CHANNELS)}
    for ch1, ch2 in pairs:
        values.append(calculate_wpli(eeg_data[channel_index[ch1]], eeg_data[channel_index[ch2]], fs))

    fz_hfd = channel_metrics["Fz"]["hfd"]
    fz_theta = channel_metrics["Fz"]["theta"]
    pz_entropy = channel_metrics["Pz"]["spectral_entropy"]
    f4_alpha = channel_metrics["F4"]["alpha"]
    f3_alpha = channel_metrics["F3"]["alpha"]

    values.append(fz_hfd / (fz_theta + 1e-6) if np.isfinite(fz_hfd) and np.isfinite(fz_theta) else np.nan)
    values.append(fz_hfd * pz_entropy if np.isfinite(fz_hfd) and np.isfinite(pz_entropy) else np.nan)
    values.append(np.log(f4_alpha + 1e-6) - np.log(f3_alpha + 1e-6)
                  if np.isfinite(f4_alpha) and np.isfinite(f3_alpha) else np.nan)

    features = np.asarray(values, dtype=float)
    if features.size != 199:
        raise RuntimeError(f"Expected 199 features, got {features.size}")
    return features


def build_feature_names() -> list[str]:
    names = [f"{ch}_{feat}" for ch in CHANNELS for feat in FEATURES_PER_CHANNEL]
    names += ["wpli_F3_F4", "wpli_P3_P4", "wpli_Fz_Pz", "wpli_O1_O2"]
    names += ["frontal_complexity_ratio", "frontal_parietal_interaction", "frontal_alpha_asymmetry"]
    assert len(names) == 199
    return names


def extract_dataset(csv_path: Path, output_path: Path, window_seconds: float = 2.0,
                    step_seconds: float = 3.0) -> None:
    df = pd.read_csv(csv_path)
    required = set(CHANNELS + ["ID", "Class"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    window_len = int(round(window_seconds * FS))
    step = int(round(step_seconds * FS))
    if window_len < FS or step < 1:
        raise ValueError("Use a window of at least one second and positive step.")

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    subject_ids: list[str] = []

    subject_order = df["ID"].drop_duplicates().tolist()
    for subject in tqdm(subject_order, desc="Extracting participants"):
        subject_df = df.loc[df["ID"] == subject]
        classes = subject_df["Class"].dropna().astype(str).unique()
        if len(classes) != 1:
            raise ValueError(f"Participant {subject} has inconsistent/missing labels: {classes}")
        label_text = classes[0].upper()
        if "CONTROL" in label_text or "NORMAL" in label_text or "NON-ADHD" in label_text:
            label = 0
        elif label_text.strip() == "ADHD" or label_text.startswith("ADHD"):
            label = 1
        else:
            raise ValueError(f"Cannot map class label {classes[0]!r} for {subject}")

        raw = subject_df[CHANNELS].to_numpy(dtype=float)
        # Same endpoint convention as the uploaded notebook: the last exact
        # window ending at len(raw) is not included.
        for start in range(0, len(raw) - window_len, step):
            epoch = raw[start:start + window_len].T
            feature_rows.append(extract_epoch_features(epoch, FS))
            labels.append(label)
            subject_ids.append(str(subject))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df = pd.DataFrame(feature_rows, columns=build_feature_names())
    feature_df.insert(0, "Class", labels)
    feature_df.insert(0, "ID", subject_ids)
    feature_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"Feature rows: {len(feature_df):,}; participants: {len(set(subject_ids))}; features: 199")
    print(feature_df["Class"].value_counts().rename(index={0: "Control", 1: "ADHD"}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Path to adhdata.csv")
    parser.add_argument("--output", type=Path, default=Path("data/features_199.csv"))
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--step-seconds", type=float, default=3.0)
    args = parser.parse_args()
    extract_dataset(args.input, args.output, args.window_seconds, args.step_seconds)


if __name__ == "__main__":
    main()
