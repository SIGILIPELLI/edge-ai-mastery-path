# Audio Wake Words & Keyword Spotting

A wake word ("Hey Siri", "Alexa", "OK Google") is a tiny always-on classifier
that listens to a microphone 24/7 on a battery-powered device and answers one
question many times a second: *was that the keyword, or not?* Everything about
this task is shaped by the "always-on" constraint — the model must be small
enough to run continuously on a microcontroller's tens-of-kilobytes of RAM,
fast enough to keep up with real-time audio, and stingy enough with power to
survive on a coin cell for months. This module builds the standard keyword
spotting (KWS) pipeline: raw audio → spectrogram features → a small
convolutional classifier → a streaming decision rule.

## Why you never feed raw audio to the model

A 1-second audio clip sampled at 16 kHz is 16,000 numbers. Feeding that
directly into a dense or even convolutional network is wasteful — audio has
structure (pitch, formants, rhythm) that lives in the *frequency* domain, not
the raw waveform. The universal first step is to convert the waveform into a
**spectrogram**: a 2D image of frequency-over-time, which convolutional
vision-style models handle very well.

The most common feature set for KWS is **MFCC** (Mel-Frequency Cepstral
Coefficients) or, in even smaller-footprint devices, straightforward Mel
**log-spectrograms**. Both start the same way: chop the audio into overlapping
windows and take the magnitude of the FFT of each one.

```python
import numpy as np
from scipy.signal import get_window

def framed_fft_features(audio, sr=16000, frame_ms=25, hop_ms=10, n_fft=512):
    """Turn a 1D waveform into a (n_frames, n_fft//2+1) power spectrogram."""
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    window = get_window("hann", frame_len)

    n_frames = 1 + (len(audio) - frame_len) // hop_len
    spec = np.empty((n_frames, n_fft // 2 + 1))
    for i in range(n_frames):
        start = i * hop_len
        frame = audio[start:start + frame_len] * window
        mag = np.abs(np.fft.rfft(frame, n=n_fft))
        spec[i] = mag ** 2  # power spectrum
    return spec
```

For a 1-second clip at 16 kHz with a 25 ms frame / 10 ms hop, that's about 98
frames of 257 frequency bins — already a `98x257` "image", but still too wide
for a tiny model. The Mel filterbank collapses those 257 linear-frequency
bins into ~40 perceptually-spaced bands.

```python
def mel_filterbank(n_fft=512, n_mels=40, sr=16000, fmin=20, fmax=8000):
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        left, center, right = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return fb

def log_mel_spectrogram(audio, sr=16000, n_mels=40):
    power_spec = framed_fft_features(audio, sr=sr)
    fb = mel_filterbank(n_fft=512, n_mels=n_mels, sr=sr)
    mel_energy = power_spec @ fb.T          # (n_frames, n_mels)
    return np.log(mel_energy + 1e-6)
```

The output — a `98x40` array for a 1-second clip — is the "image" a small CNN
will classify. Full MFCC adds a Discrete Cosine Transform on top of this to
decorrelate the bands; many microcontroller KWS models skip that step and
train directly on log-mel energies, since the CNN's convolutions can learn
the decorrelation themselves.

## The model: tiny CNN over the spectrogram

Architectures like Google's original "DS-CNN" (depthwise-separable CNN) or
the simpler few-layer CNNs used in TensorFlow's `micro_speech` example
follow the same recipe: 2–4 conv layers with small channel counts, a global
pool, then a dense layer to `n_keywords + 1` classes (the `+1` is a
"background/unknown" class — critical, since the device hears far more
non-keyword audio than keyword audio).

```python
# Described, not run: this needs a Keras/TF training environment.
# model = keras.Sequential([
#     keras.layers.Input(shape=(98, 40, 1)),
#     keras.layers.Conv2D(8, (10, 4), strides=(2, 2), activation="relu"),
#     keras.layers.Conv2D(16, (3, 3), strides=(1, 1), activation="relu"),
#     keras.layers.GlobalAveragePooling2D(),
#     keras.layers.Dense(len(labels), activation="softmax"),
# ])
```

This class of model is commonly 15-25K parameters, quantizes to a 15-30 KB
`.tflite` file, and runs an inference in a few milliseconds on a Cortex-M4 —
well inside the "always-on" power budget when duty-cycled correctly (next
section).

## Streaming inference: sliding window, not one-shot

A real device doesn't get a neatly-clipped 1-second recording — it gets a
continuous audio stream. The standard approach is a **sliding window**: keep
a rolling 1-second buffer, run the classifier every 200-500 ms on the current
window contents, and require **N consecutive detections** above a confidence
threshold before firing the wake event. This "debounce" logic trades a little
latency for a large reduction in false positives, since a single noisy frame
misclassification won't trigger anything on its own.

```python
class StreamingDetector:
    def __init__(self, threshold=0.85, consecutive_needed=2):
        self.threshold = threshold
        self.consecutive_needed = consecutive_needed
        self._hits = 0

    def update(self, keyword_prob):
        """Call once per inference window; returns True on a firm detection."""
        if keyword_prob >= self.threshold:
            self._hits += 1
        else:
            self._hits = 0
        return self._hits >= self.consecutive_needed
```

## Edge-AI tradeoffs

**Window length vs. latency.** A longer analysis window (1.5s) captures more
context and cuts false positives, but adds perceived lag before the device
"wakes up." Most commercial wake-word engines land near 1 second as the
sweet spot.

**Sampling rate vs. compute.** 16 kHz is standard for speech; dropping to
8 kHz halves the FFT and feature-extraction cost but throws away frequencies
above 4 kHz, which hurts fricative sounds ("s", "f") that distinguish similar
words.

**Always-on power vs. detection latency.** Running the full pipeline every
10 ms is expensive; many real chips run a cheap energy/voice-activity-detector
first (a simple RMS-threshold on raw audio, near-zero cost) and only invoke
the spectrogram + CNN pipeline when sound is present. This can cut average
power by 10x at the cost of a few milliseconds of extra latency on
wake-word onset.

**False accepts vs. false rejects.** Lowering the detection threshold makes
the device more responsive but triggers on TV audio and other speech; raising
it makes users repeat themselves. Tune this against a large recorded set of
"hard negatives" — audio from the deployment environment that is *not* the
keyword — never just against clean lab recordings.

## Cheat sheet

| Concept | Typical value / choice |
|---|---|
| Sample rate | 16 kHz (speech), 8 kHz on very tight budgets |
| Frame / hop length | 25 ms / 10 ms |
| Mel bands | 32–40 |
| Spectrogram shape (1s clip) | ~98 x 40 |
| Model size | 15–25K params, 15–30 KB int8 `.tflite` |
| Classes | keywords + "unknown" + "silence/background" |
| Sliding window | 1.0–1.5 s, re-run every 200–500 ms |
| Debounce | require 2–3 consecutive positive windows |
| Power trick | cheap VAD gate before running the CNN |
| Latency budget | detection within ~300-500 ms of keyword end |

## Exercise

1. Implement `framed_fft_features` and `log_mel_spectrogram` above (they're
   pure numpy/scipy — no TensorFlow needed) and run them on a synthetic
   signal: a 1-second 440 Hz sine wave plus white noise. Plot the resulting
   log-mel spectrogram and confirm you see energy concentrated near the band
   containing 440 Hz.
2. Change `n_mels` from 40 to 13 and to 80. How does the spectrogram's shape
   and visual detail change? What's the tradeoff for a downstream CNN?
3. Implement `StreamingDetector` and simulate a sequence of keyword
   probabilities like `[0.2, 0.3, 0.9, 0.92, 0.4, 0.88, 0.91, 0.93]` at
   `consecutive_needed=2` and `=3`. At which index does each configuration
   fire, and why does the difference matter for false-accept rate?
4. Describe (in prose, referencing the TFLite quantization workflow from
   Level 1 Module 05) the full path from a trained Keras KWS model to a
   deployable int8 `.tflite` file, including what a representative dataset
   for this task would need to contain.
