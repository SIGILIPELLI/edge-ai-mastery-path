# 03 · Training a Tiny Model

Time to run stage 1 of the pipeline. We'll train the "hello world" of
TinyML: a tiny neural network that learns **sin(x)** from noisy samples.
It sounds trivial, but it's the standard first model (it's the official
TFLite-Micro hello-world example) for good reasons: it trains in seconds on
any laptop, its correctness is visible at a glance, and it's small enough
that in Module 06 you'll read its entire C array by eye. Everything here is
plain Keras — the edge-specific discipline is *keeping the model tiny*.

## Setup

You need Python 3.9+ and TensorFlow. In a fresh virtual environment:

```bash
python3 -m venv tinyml-env
source tinyml-env/bin/activate        # Windows: tinyml-env\Scripts\activate
pip install tensorflow numpy matplotlib
python -c "import tensorflow as tf; print(tf.__version__)"
```

Any recent TensorFlow (2.x) works; no GPU needed — this model trains in
under a minute on a CPU.

## The dataset: noisy sine samples

We generate it ourselves — no downloads. Inputs are x values in
[0, 2π]; targets are sin(x) plus a little noise, because real sensor data is
never clean:

```python
import numpy as np

rng = np.random.default_rng(42)

SAMPLES = 1000
x = rng.uniform(0.0, 2 * np.pi, SAMPLES).astype(np.float32)
y = (np.sin(x) + rng.normal(0, 0.1, SAMPLES)).astype(np.float32)

# Shuffle, then split 60/20/20 into train/validation/test
idx = rng.permutation(SAMPLES)
x, y = x[idx], y[idx]
n_train, n_val = 600, 200
x_train, y_train = x[:n_train], y[:n_train]
x_val,   y_val   = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
x_test,  y_test  = x[n_train + n_val:], y[n_train + n_val:]

print(x_train.shape, x_val.shape, x_test.shape)   # (600,) (200,) (200,)
```

The three-way split matters even for a toy: **train** fits the weights,
**validation** watches for overfitting during training, and **test** is
touched exactly once at the end — and will be reused in Modules 04–05 to
verify the converted and quantized models.

## The model: 321 parameters

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()
```

The summary reports **321 trainable parameters** — (1×16+16) + (16×16+16) +
(16×1+1). As float32 that's ~1.3 KB of weights. Two hidden layers of 16
ReLU units is plenty to bend a line into a sine curve; a single Dense(1)
with no hidden layers would only manage a straight line (try it — that's
exercise 2).

!!! tip "Think in parameters from day one"
    `model.summary()` is the most important line in edge-model training.
    Params × 1 byte ≈ your eventual int8 flash cost. Get in the habit of
    reading it before you ever hit `fit`.

## Training

```python
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=300,
    batch_size=32,
    verbose=0,
)
loss, mae = model.evaluate(x_test, y_test, verbose=0)
print(f"test MSE={loss:.4f}  test MAE={mae:.4f}")
```

Expect a test MAE around **0.08–0.10** — you can't do much better, because
we injected noise with a standard deviation of 0.1. A model beating the
noise floor would be memorizing noise, not learning the function.

Plot predictions against the truth to *see* the fit:

```python
import matplotlib.pyplot as plt

x_plot = np.linspace(0, 2 * np.pi, 200, dtype=np.float32)
y_pred = model.predict(x_plot.reshape(-1, 1), verbose=0)

plt.scatter(x_test, y_test, s=8, alpha=0.4, label="test data")
plt.plot(x_plot, np.sin(x_plot), "g-", label="true sin(x)")
plt.plot(x_plot, y_pred, "r-", label="model")
plt.legend(); plt.savefig("fit.png")
```

The red curve should hug the green one across the whole range. If it's a
straight line, training diverged or the model is too small; if it wiggles
through individual points, it's overfitting the noise.

## Saving the model — and freezing test data for later

```python
model.save("sine_model.keras")

# Save the test set: Modules 04-05 reuse EXACTLY these points
np.savez("sine_test_data.npz", x_test=x_test, y_test=y_test)
```

`sine_model.keras` (a full Keras save: architecture + weights + optimizer
state) is the artifact that crosses the first arrow of the pipeline — the
converter in Module 04 consumes it. Saving the test set alongside it is the
"verify every stage" rule in action: identical inputs at every stage, so any
output difference is attributable to the stage, never to the data.

## Cheat sheet

| Task | Code |
|---|---|
| Define tiny MLP | `keras.Sequential([...Dense(16, activation="relu")...])` |
| Count parameters | `model.summary()` / `model.count_params()` |
| Compile for regression | `model.compile(optimizer="adam", loss="mse", metrics=["mae"])` |
| Train with validation | `model.fit(x, y, validation_data=(xv, yv), epochs=N)` |
| Final unbiased score | `model.evaluate(x_test, y_test)` |
| Predict | `model.predict(x.reshape(-1, 1))` |
| Save model | `model.save("sine_model.keras")` |
| Save arrays | `np.savez("f.npz", a=a, b=b)` / `np.load("f.npz")` |
| Reproducibility | `np.random.default_rng(seed)`, fixed splits |

## Exercise

1. Run the full script and report your test MAE. Then retrain with
   `Dense(8)`/`Dense(8)` hidden layers and with `Dense(64)`/`Dense(64)`.
   Record parameters and test MAE for all three in a small table — where do
   extra parameters stop paying?
2. Delete both hidden layers (leaving one `Dense(1)`) and retrain. Plot the
   result and explain in one sentence *why* the shape is what it is.
3. Change the data range from [0, 2π] to [0, 4π] and retrain the original
   model. Does 16+16 still fit two full periods well? What's the smallest
   architecture you can find that keeps test MAE under 0.12?
4. (Prep for Module 04) Confirm `sine_model.keras` and
   `sine_test_data.npz` exist and reload both in a fresh Python session —
   `keras.models.load_model(...)` should evaluate to the same test MAE.
