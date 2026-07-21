# 08 · Sensors & Feature Extraction

So far our models ate single numbers. Real edge AI eats **sensor streams**
— an accelerometer emitting 100 (x, y, z) readings per second, a microphone
emitting 16,000 samples per second — and streams don't fit a model's fixed
input size. The bridge is **feature extraction**: slicing the stream into
windows and transforming each window into a compact, informative vector.
This is the least glamorous and most decisive part of every TinyML system —
teams routinely gain more accuracy from better features than from bigger
models. Everything in this module runs in NumPy on your laptop; every
technique translates directly to C on-device.

## Windowing: from stream to samples

A model needs a fixed-size input, so we chop the stream into fixed-length
**windows**, usually overlapping so events near window edges aren't missed:

```text
stream:   ────────────────────────────────────────────►  time
window 1: [══════════ 2 s ══════════]
window 2:            [══════════ 2 s ══════════]           50% overlap
window 3:                       [══════════ 2 s ══════════]
```

```python
import numpy as np

def make_windows(stream, window_len, hop):
    """stream: (n_samples, n_channels) -> (n_windows, window_len, n_channels)"""
    wins = []
    for start in range(0, len(stream) - window_len + 1, hop):
        wins.append(stream[start:start + window_len])
    return np.stack(wins)

# 10 s of fake 3-axis accelerometer data at 100 Hz
rng = np.random.default_rng(1)
stream = rng.normal(0, 1, size=(1000, 3)).astype(np.float32)

windows = make_windows(stream, window_len=200, hop=100)  # 2 s windows, 50% overlap
print(windows.shape)   # (9, 200, 3)
```

Two design decisions hide in those arguments. **Window length**: long
enough to contain the event (a gesture ≈ 1–2 s; a spoken keyword ≈ 1 s).
**Hop**: smaller hop = more frequent decisions and more compute. On-device,
the same idea is implemented as a ring buffer that always holds the most
recent `window_len` samples.

## Time-domain features: cheap and surprisingly strong

For many motion tasks you don't feed raw samples to the model at all — you
compute a handful of statistics per window per axis:

```python
def time_features(win):
    """win: (window_len, n_channels) -> flat feature vector"""
    feats = []
    for ch in range(win.shape[1]):
        s = win[:, ch]
        feats += [
            s.mean(),                          # DC level / orientation
            s.std(),                           # overall variability
            np.sqrt(np.mean(s ** 2)),          # RMS: energy of the motion
            s.min(), s.max(),
            np.mean(np.abs(np.diff(s))),       # jerkiness
        ]
    return np.array(feats, dtype=np.float32)

fv = time_features(windows[0])
print(fv.shape)    # (18,) = 6 features x 3 axes
```

Eighteen floats instead of 600 raw values: the downstream model shrinks
from thousands of weights to hundreds, and every one of these features is
a few lines of integer-friendly C. **RMS** (root-mean-square) deserves
special mention — it's the single best "how much is happening?" number, and
alone it separates *still* from *shaking* almost perfectly (the capstone
exploits this).

## Frequency-domain features: the FFT

Repetitive signals — vibration, audio, walking — are described far better
by *which frequencies* they contain than by their raw waveform. The Fast
Fourier Transform decomposes a window into its frequency components:

```python
def fft_features(win_1ch, sample_rate=100, n_bins=8):
    """One channel -> energy in n_bins frequency bands."""
    spectrum = np.abs(np.fft.rfft(win_1ch))          # magnitude spectrum
    spectrum = spectrum[1:]                           # drop DC
    bands = np.array_split(spectrum, n_bins)          # group into coarse bins
    return np.array([b.mean() for b in bands], dtype=np.float32)

# A 5 Hz vibration vs. flat noise — trivially separable in frequency space
t = np.arange(200) / 100.0
vib   = np.sin(2 * np.pi * 5 * t) + rng.normal(0, 0.2, 200)
noise = rng.normal(0, 0.5, 200)
print(fft_features(vib).round(1))    # energy spike in the low bins
print(fft_features(noise).round(1))  # roughly flat
```

Binning the spectrum into ~8–32 coarse bands keeps the feature vector tiny
while preserving the shape of the spectrum — this is exactly what
commercial predictive-maintenance sensors ship. Two facts to carry to
Level 2: an FFT of a real signal of length N yields N/2+1 frequencies up to
half the sample rate (Nyquist), and MCU DSP libraries (CMSIS-DSP, ESP-DSP)
provide fixed-point FFTs fast enough to run continuously.

## Audio: spectrograms, conceptually

Audio classification stacks FFTs: slice 1 s of audio into ~30–50 ms
sub-frames, FFT each, and lay the resulting spectra side by side to form a
**spectrogram** — a 2-D "image" of time × frequency. Speech models compress
the frequency axis with a mel filterbank (perceptually spaced bands) to get
**MFCCs/mel-spectrograms**, typically ~40 bands × ~50 frames — at which
point keyword spotting becomes small-image classification, handled by the
same tiny CNNs you already know. Level 2's wake-word module builds this
end-to-end; for now the takeaway is that *audio pipelines are windowing +
FFT applied twice*.

## The cardinal rule: identical preprocessing on both sides

Your features are computed twice in every deployed system — in Python
during training, and in C on-device forever after. **Any mismatch (window
length, normalization, scaling, bin edges, sample rate) silently destroys
accuracy** while everything appears to work. Defenses, in order of value:

1. Write preprocessing as small pure functions (as above) mirroring what C
   can do — no exotic library calls.
2. Save a few windows and their Python-computed features to a file; make
   your C implementation reproduce them in a desktop unit test
   (Module 06 skills) before flashing anything.
3. Log a feature vector from the device and diff against Python on the
   same raw window whenever accuracy mysteriously drops (Module 09).

## Cheat sheet

| Concept | Essentials |
|---|---|
| Windowing | fixed `window_len`, overlapping via `hop`; ring buffer on-device |
| Window length | must contain the event (gesture 1–2 s, keyword ~1 s) |
| Mean | orientation / DC level |
| RMS | `sqrt(mean(s**2))` — energy; the workhorse motion feature |
| Std dev | variability around the mean |
| Mean abs diff | jerkiness / smoothness |
| FFT | `np.abs(np.fft.rfft(s))` — magnitude spectrum |
| Usable band | 0 … sample_rate/2 (Nyquist) |
| Coarse spectrum | average spectrum into 8–32 bins |
| Spectrogram | FFTs of sub-frames stacked over time (audio → image) |
| Rule zero | train-time and device preprocessing must match exactly |

## Exercise

1. Generate three synthetic 3-axis "activities" (still: tiny noise; walk:
   2 Hz sine + noise; shake: 8 Hz sine, larger amplitude), window them, and
   compute `time_features` for each window. Print per-class mean RMS —
   how well does RMS alone separate the classes?
2. Train a tiny Keras classifier (one Dense(16) hidden layer) on those
   18-dim feature vectors, three classes. Report test accuracy — then
   re-run the whole pipeline (convert → int8 → verify) from Modules 04–05
   on it. You've just done the full workflow on multichannel sensor data.
3. Compute `fft_features` for walk vs. shake windows and identify which
   bin peaks for each. Does the peak bin match the sine frequencies you
   injected? Show the arithmetic (bin width = Nyquist / n_bins).
4. Break rule zero on purpose: train on windows normalized with
   `(s - mean) / std` but evaluate on unnormalized windows. Measure the
   accuracy drop, and write one sentence on why nothing "errored."
