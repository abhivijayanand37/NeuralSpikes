"""
ALL-CHANNEL AVERAGE version of the raw-only figure
----------------------------------------------------------------
Runs the same analysis as raw_only_comparison_figure.py on EVERY extracellular
channel (0-5) of one recording, then averages the results across channels.

For each channel:
  1. the threshold multiplier k is tuned on that channel's TRAINING half only,
  2. that single k is applied once to the held-out TEST half.
Then the per-channel test scores are averaged (mean +/- SD across channels).

Works for raw OR a bandpass-filtered signal, so you can make the matching
pair of figures and compare them by hand:

  python all_channels_average.py data/d18011.001.dat              (raw)
  python all_channels_average.py data/d18011.001.dat 200-3000     (200-3000 Hz band)
  python all_channels_average.py data/d18011.002.dat              (second recording, raw)

Same settings as before: k = 2 to 11.5 in steps of 0.5, refractory = 80
samples, 20-sample (1 ms) matching tolerance, 50/50 train/test split.

IMPORTANT: channels from one recording share the same intracellular ground
truth, so they are NOT independent samples. This script gives the average
WITHIN one recording. To report results across the study, carry each
recording's MEAN row (from the CSV) forward and average those across
recordings; the recording is the unit of replication.

Saves (in the current folder):
  <recording>_all_channels_<raw or band>_average.png
  <recording>_all_channels_<raw or band>_average.csv   (per channel + MEAN + SD rows)
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/d18011.005.dat"
RECORDING = os.path.splitext(os.path.basename(DATA_PATH))[0]

# Condition: raw by default, or a band like "200-3000" given as the 2nd argument
if len(sys.argv) > 2 and sys.argv[2].lower() != "raw":
    low, high = sys.argv[2].split("-")
    BAND = (float(low), float(high))
    LABEL = f"{int(BAND[0])}-{int(BAND[1])}Hz"
    PRETTY = f"Bandpass {int(BAND[0])}-{int(BAND[1])} Hz"
    COLOR = "#2a78d6"                   # blue for filtered
else:
    BAND = None
    LABEL = "raw"
    PRETTY = "Raw (unfiltered)"
    COLOR = "#eb6834"                   # orange for raw

FS = 20000
TOLERANCE = 20
GT_K = 10
REFRACTORY = 80
K_VALUES = np.arange(2, 12, 0.5)
CHANNELS = list(range(6))               # extracellular channels 0-5

INK = "#0b0b0b"
INK2 = "#52514e"

# To compare two figures by eye, set the SAME limits when you make each one,
# for example YLIM_F1_VS_K = (0, 0.06) and YLIM_TEST_BARS = (0, 0.6). None = auto.
YLIM_F1_VS_K = None
YLIM_TEST_BARS = None
YLIM_PER_CHANNEL = None


# ============================================================
# Load data and split into train / test halves
# ============================================================

data = np.fromfile(DATA_PATH, dtype=np.int16).reshape(-1, 8)
split_idx = int(len(data) * 0.5)
test_seconds = (len(data) - split_idx) / FS

train_intra = data[:split_idx, 6].astype(np.float64)
test_intra = data[split_idx:, 6].astype(np.float64)


# ============================================================
# Helper functions (same logic as the other scripts)
# ============================================================

def bandpass_filter(signal, lowcut, highcut, fs=FS, order=4):
    nyquist = fs / 2
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, signal)


def noise_stats(signal):
    median = np.median(signal)
    mad = np.median(np.abs(signal - median)) / 0.6745
    return median, mad


def detect_spikes(signal, threshold, refractory_period=80, direction="below"):
    if direction == "above":
        crossings = np.flatnonzero(signal > threshold)
    else:
        crossings = np.flatnonzero(signal < threshold)
    detected = []
    pos = 0
    n = len(crossings)
    while pos < n:
        idx = crossings[pos]
        detected.append(idx)
        pos = np.searchsorted(crossings, idx + refractory_period, side="left")
    return np.array(detected, dtype=np.int64)


def _within_tolerance(a, b, tolerance):
    idx = np.searchsorted(b, a)
    left = b[np.clip(idx - 1, 0, len(b) - 1)]
    right = b[np.clip(idx, 0, len(b) - 1)]
    return np.minimum(np.abs(a - left), np.abs(a - right)) <= tolerance


def score(detected, true_spikes, tolerance=TOLERANCE):
    """Returns precision, recall, true spikes caught, detections that were correct."""
    if len(detected) == 0 or len(true_spikes) == 0:
        return 0.0, 0.0, 0, 0
    matched_detected = _within_tolerance(detected, true_spikes, tolerance)
    matched_true = _within_tolerance(true_spikes, detected, tolerance)
    precision = matched_detected.sum() / len(detected)
    recall = matched_true.sum() / len(true_spikes)
    return precision, recall, int(matched_true.sum()), int(matched_detected.sum())


def f1_score(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ============================================================
# Ground truth (same for every channel in this recording)
# ============================================================

mi, si = noise_stats(train_intra)
train_true = detect_spikes(train_intra, mi + GT_K * si, REFRACTORY, "above")
mi, si = noise_stats(test_intra)
test_true = detect_spikes(test_intra, mi + GT_K * si, REFRACTORY, "above")
print(f"Recording {RECORDING} | condition: {PRETTY}")
print(f"True spikes - train: {len(train_true)}, test: {len(test_true)}\n")


# ============================================================
# Run every channel: tune k on TRAIN, score once on TEST
# ============================================================

per_channel = []
f1_curves = []

for ch in CHANNELS:
    train_sig = data[:split_idx, ch].astype(np.float64)
    test_sig = data[split_idx:, ch].astype(np.float64)
    if BAND is not None:
        train_sig = bandpass_filter(train_sig, *BAND)
        test_sig = bandpass_filter(test_sig, *BAND)

    median, mad = noise_stats(train_sig)
    rows = []
    for k in K_VALUES:
        detected = detect_spikes(train_sig, median - k * mad, REFRACTORY, "below")
        p, r, tp_true, tp_det = score(detected, train_true)
        rows.append((k, p, r, f1_score(p, r)))
    f1_curves.append([row[3] for row in rows])

    best_k, tr_p, tr_r, tr_f1 = max(rows, key=lambda row: row[3])

    t_median, t_mad = noise_stats(test_sig)
    test_detected = detect_spikes(test_sig, t_median - best_k * t_mad, REFRACTORY, "below")
    te_p, te_r, te_caught, te_tp_det = score(test_detected, test_true)
    te_f1 = f1_score(te_p, te_r)
    false_alarms = (len(test_detected) - te_tp_det) / test_seconds

    per_channel.append({
        "channel": ch, "best_k": best_k,
        "train_precision": tr_p, "train_recall": tr_r, "train_f1": tr_f1,
        "test_precision": te_p, "test_recall": te_r, "test_f1": te_f1,
        "test_caught": te_caught, "test_true_spikes": len(test_true),
        "test_false_alarms_per_s": false_alarms,
    })
    print(f"ch {ch}: best k={best_k:<5g} | test precision={te_p:.4f} recall={te_r:.4f} "
          f"F1={te_f1:.4f} | caught {te_caught}/{len(test_true)} | {false_alarms:.2f} false alarms/s")

f1_curves = np.array(f1_curves)                     # channels x k
metrics = ["test_precision", "test_recall", "test_f1", "test_false_alarms_per_s"]
means = {m: np.mean([c[m] for c in per_channel]) for m in metrics}
sds = {m: np.std([c[m] for c in per_channel], ddof=1) if len(per_channel) > 1 else 0.0 for m in metrics}

print(f"\nMEAN +/- SD across {len(CHANNELS)} channels (held-out test):")
for m in metrics:
    print(f"  {m:<26} {means[m]:.4f} +/- {sds[m]:.4f}")


# ============================================================
# Save CSV (per channel + MEAN + SD rows)
# ============================================================

csv_name = f"{RECORDING}_all_channels_{LABEL}_average.csv"
fields = ["recording", "condition", "channel", "best_k", "train_precision", "train_recall", "train_f1",
          "test_precision", "test_recall", "test_f1", "test_caught", "test_true_spikes",
          "test_false_alarms_per_s"]
with open(csv_name, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for c in per_channel:
        writer.writerow({"recording": RECORDING, "condition": PRETTY, **c})
    for name, stat in (("MEAN", np.mean), ("SD", lambda v: np.std(v, ddof=1) if len(v) > 1 else 0.0)):
        row = {"recording": RECORDING, "condition": PRETTY, "channel": name}
        for key in fields[3:]:
            row[key] = stat([c[key] for c in per_channel])
        writer.writerow(row)
print(f"\nSaved data: {csv_name}")


# ============================================================
# Figure (three labeled panels)
# ============================================================

n_ch = len(CHANNELS)
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
fig.suptitle(f"{PRETTY}: average of {n_ch} channels - recording {RECORDING}",
             fontsize=14, color=INK, x=0.01, ha="left")

# Panel 1: F1 vs k, every channel (thin) and the mean (thick)
ax = axes[0]
for i, curve in enumerate(f1_curves):
    ax.plot(K_VALUES, curve, color=COLOR, alpha=0.3, linewidth=1,
            label="Individual channels" if i == 0 else None)
mean_curve = f1_curves.mean(axis=0)
ax.plot(K_VALUES, mean_curve, color=COLOR, marker="o", linewidth=2.5, label="Mean across channels")
ax.set_title("F1 vs. k (Training)")
ax.set_xlabel("k (threshold = median - k*MAD)")
ax.set_ylabel("F1 Score")
if YLIM_F1_VS_K:
    ax.set_ylim(*YLIM_F1_VS_K)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Panel 2: mean held-out test scores with SD bars and per-channel dots
ax = axes[1]
names = ["Precision", "Recall", "F1"]
keys = ["test_precision", "test_recall", "test_f1"]
x = np.arange(3)
mean_vals = [means[k] for k in keys]
sd_vals = [sds[k] for k in keys]
bars = ax.bar(x, mean_vals, yerr=sd_vals, capsize=6, color=COLOR, width=0.55,
              label=f"Mean +/- SD across {n_ch} channels", error_kw={"ecolor": INK2, "linewidth": 1.2})
rng = np.random.default_rng(0)
for xi, key in zip(x, keys):
    vals = [c[key] for c in per_channel]
    jitter = rng.uniform(-0.12, 0.12, len(vals))
    ax.scatter(xi + jitter, vals, color=INK, s=14, zorder=3,
               label="Individual channels" if xi == 0 else None)
for xi, mv, sv in zip(x, mean_vals, sd_vals):
    ax.annotate(f"{mv:.4f}", (xi + 0.3, mv), ha="left", va="center", fontsize=10, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_title("Held-Out Test (mean across channels)")
ax.set_ylabel("Score")
if YLIM_TEST_BARS:
    ax.set_ylim(*YLIM_TEST_BARS)
else:
    ax.set_ylim(bottom=0)               # scores can't be negative; SD bars shouldn't suggest otherwise
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Panel 3: held-out test F1 for each channel
ax = axes[2]
chs = [f"ch {c['channel']}" for c in per_channel]
f1_vals = [c["test_f1"] for c in per_channel]
ax.bar(chs, f1_vals, color=COLOR, width=0.6, label="Test F1 per channel")
ax.axhline(means["test_f1"], color=INK2, linestyle="--", linewidth=1, label=f"Mean = {means['test_f1']:.4f}")
for i, c in enumerate(per_channel):
    ax.annotate(f"{c['test_caught']}/{c['test_true_spikes']}", (i, c["test_f1"]),
                ha="center", va="bottom", fontsize=9, color=INK2, xytext=(0, 3), textcoords="offset points")
ax.set_title("Held-Out Test F1 by Channel\n(label = true spikes caught / total)")
ax.set_ylabel("F1 Score")
if YLIM_PER_CHANNEL:
    ax.set_ylim(*YLIM_PER_CHANNEL)
else:
    ax.set_ylim(0, max(f1_vals + [means["test_f1"]]) * 1.25 + 1e-9)   # headroom for the labels
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout(rect=(0, 0, 1, 0.94))
png_name = f"{RECORDING}_all_channels_{LABEL}_average.png"
plt.savefig(png_name, dpi=150)
print(f"Saved figure: {png_name}")