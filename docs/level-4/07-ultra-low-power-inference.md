# Ultra-Low-Power Inference

Every optimization so far (quantization, pruning, NPU offload, compiler
fusion) has targeted latency or accuracy. For battery- or energy-
harvesting-powered devices — a coin-cell sensor node meant to run for
years, an energy-harvesting tag with no battery at all — the metric that
actually determines whether the product works is **energy per inference**,
and the biggest lever on that number usually isn't the model at all: it's
how aggressively the device sleeps between inferences. This module covers
duty-cycling design and the always-on/wake-on-event pattern, with a
tested Python energy-budget simulator, since actual microamp-level power
measurement needs a power analyzer and real silicon this environment
doesn't have.

## The energy budget, decomposed

Total energy for a battery-powered inference device over its target
lifetime breaks into three components that trade off against each other:

```
E_total = E_sleep * t_sleep + E_active * t_active + E_wake_transition * n_wakes
```

Where `E_sleep` is often 1,000-10,000x smaller than `E_active` on modern
microcontrollers (deep sleep modes can pull under 1 µA vs. several mA
active), which means **how much of the device's life is spent asleep**
dominates the total energy calculation far more than how efficient the
active-mode inference itself is — a genuinely different optimization
target than everything in Levels 1-3, which implicitly assumed the device
was always powered and always available to run inference on demand.

## Simulating the energy tradeoff

```python
def simulate_energy_budget(duty_cycle_hz, active_duration_ms,
                            sleep_current_ua, active_current_ma,
                            wake_transition_ua_ms, supply_voltage=3.3,
                            duration_hours=24):
    """Computes total energy (in milliwatt-hours) for a device that wakes
    up duty_cycle_hz times per second, runs inference for
    active_duration_ms each time, and sleeps otherwise. wake_transition
    models the fixed micro-amp-millisecond cost of the sleep->active
    transition itself (clock startup, sensor power-up), which is often
    non-negligible at high duty cycles."""
    total_seconds = duration_hours * 3600
    n_wakes = duty_cycle_hz * total_seconds
    active_seconds = n_wakes * (active_duration_ms / 1000)
    sleep_seconds = total_seconds - active_seconds

    sleep_energy_uas = sleep_current_ua * sleep_seconds          # microamp-seconds
    active_energy_uas = (active_current_ma * 1000) * active_seconds
    wake_energy_uas = wake_transition_ua_ms * n_wakes / 1000     # convert ua*ms -> ua*s

    total_uas = sleep_energy_uas + active_energy_uas + wake_energy_uas
    total_mwh = (total_uas / 1000) * supply_voltage / 3600       # uAs->mAs->mWh
    return {
        "n_wakes": int(n_wakes),
        "active_time_pct": round(active_seconds / total_seconds * 100, 4),
        "sleep_energy_mwh": round(sleep_energy_uas / 1000 * supply_voltage / 3600, 4),
        "active_energy_mwh": round(active_energy_uas / 1000 * supply_voltage / 3600, 4),
        "wake_transition_energy_mwh": round(wake_energy_uas / 1000 * supply_voltage / 3600, 4),
        "total_mwh_per_day": round(total_mwh, 4),
    }


# Keyword-spotting node: wakes 1x/sec, 15ms inference, deep-sleep 2uA.
result_1hz = simulate_energy_budget(
    duty_cycle_hz=1, active_duration_ms=15,
    sleep_current_ua=2.0, active_current_ma=8.0, wake_transition_ua_ms=500)

# Same model, but wakes 10x/sec instead (e.g. lower-latency requirement).
result_10hz = simulate_energy_budget(
    duty_cycle_hz=10, active_duration_ms=15,
    sleep_current_ua=2.0, active_current_ma=8.0, wake_transition_ua_ms=500)

print("1 Hz duty cycle:", result_1hz)
print("10 Hz duty cycle:", result_10hz)

typical_coin_cell_mwh = 3.0 * 220   # CR2032: ~3V, ~220mAh -> mWh
print(f"\nCR2032 coin cell capacity: ~{typical_coin_cell_mwh:.0f} mWh")
print(f"estimated lifetime at 1 Hz: {typical_coin_cell_mwh / result_1hz['total_mwh_per_day']:.0f} days")
print(f"estimated lifetime at 10 Hz: {typical_coin_cell_mwh / result_10hz['total_mwh_per_day']:.0f} days")
```

Running this prints:

```
1 Hz duty cycle: {'n_wakes': 86400, 'active_time_pct': 1.5, 'sleep_energy_mwh': 0.156, 'active_energy_mwh': 9.504, 'wake_transition_energy_mwh': 0.0396, 'total_mwh_per_day': 9.6996}
10 Hz duty cycle: {'n_wakes': 864000, 'active_time_pct': 15.0, 'sleep_energy_mwh': 0.1346, 'active_energy_mwh': 95.04, 'wake_transition_energy_mwh': 0.396, 'total_mwh_per_day': 95.5706}

CR2032 coin cell capacity: ~660 mWh
estimated lifetime at 1 Hz: 68 days
estimated lifetime at 10 Hz: 7 days
```

With these particular parameters, `active_energy_mwh` dominates the total
in both cases (9.5 of 9.7 mWh/day at 1 Hz; 95.0 of 95.6 at 10 Hz) —
`sleep_energy_mwh` and `wake_transition_energy_mwh` are both negligible
here because the assumed 2 µA sleep current and 500 µA·ms wake transition
are simply small relative to running an 8 mA active inference for 15 ms.
The 10x increase in duty cycle produces almost exactly a 10x increase in
total daily energy (9.7 -> 95.6 mWh/day) because active energy scales
linearly with wake count — the practical lesson from *this* parameter
regime is that active-mode energy (and therefore active duration and
current draw — the things Levels 1-3's model optimizations directly
affect) is the lever that matters most, while sleep current and wake
overhead only start to dominate once they're pushed low enough, or the
duty cycle low enough, to make active energy the smaller term. Try
reducing `sleep_current_ua` and `wake_transition_ua_ms` further, or
lowering the duty cycle toward true event-driven rates, to see the
crossover — which is exactly what the next section does.

## Wake-on-event: moving the "when to wake" decision to cheaper hardware

The most effective ultra-low-power pattern skips periodic duty-cycling
entirely: a cheap, always-on analog or digital front-end (a simple
threshold comparator on a microphone's amplitude, a PIR sensor's
hardware interrupt line) wakes the main MCU only when something worth
inferring about actually happens, rather than waking on a fixed timer and
often finding nothing.

```python
def simulate_wake_on_event(events_per_hour, active_duration_ms,
                            sleep_current_ua, active_current_ma,
                            wake_transition_ua_ms, always_on_sensor_current_ua,
                            supply_voltage=3.3, duration_hours=24):
    """Same energy model as before, but wakes are driven by actual events
    (events_per_hour) rather than a fixed duty cycle, at the cost of a
    small continuously-running sensor front-end that must stay powered
    to detect those events at all."""
    total_seconds = duration_hours * 3600
    n_wakes = events_per_hour * duration_hours
    active_seconds = n_wakes * (active_duration_ms / 1000)
    sleep_seconds = total_seconds - active_seconds

    sleep_energy_uas = sleep_current_ua * sleep_seconds
    active_energy_uas = (active_current_ma * 1000) * active_seconds
    wake_energy_uas = wake_transition_ua_ms * n_wakes / 1000
    always_on_energy_uas = always_on_sensor_current_ua * total_seconds

    total_uas = sleep_energy_uas + active_energy_uas + wake_energy_uas + always_on_energy_uas
    total_mwh = (total_uas / 1000) * supply_voltage / 3600
    return {"n_wakes": int(n_wakes), "total_mwh_per_day": round(total_mwh, 4)}


# Same node, but only wakes on genuine acoustic events (say ~20/hour
# instead of a fixed 86,400 wakes/day at 1Hz), plus a 3uA always-on
# comparator front-end watching the microphone continuously.
result_event = simulate_wake_on_event(
    events_per_hour=20, active_duration_ms=15,
    sleep_current_ua=2.0, active_current_ma=8.0,
    wake_transition_ua_ms=500, always_on_sensor_current_ua=3.0)

print("wake-on-event:", result_event)
print(f"estimated lifetime: {typical_coin_cell_mwh / result_event['total_mwh_per_day']:.0f} days")
```

Running this prints:

```
wake-on-event: {'n_wakes': 480, 'total_mwh_per_day': 0.449}
estimated lifetime: 1470 days
```

Going from 86,400 timer-driven wakes/day down to 480 genuine event-driven
wakes/day (a ~180x reduction) extends estimated lifetime from 68 days to
roughly 4 years — a difference in kind, not degree, and one that came
entirely from changing *when* the device wakes (plus paying a small
continuous 3 µA cost for the always-on trigger sensor), without touching
the model or the inference code at all. This is why ultra-low-power edge
AI projects almost always start with the wake-trigger design, not model
optimization — once wake frequency drops far enough, the active-energy
term that dominated the duty-cycle scenario above shrinks to the point
where it's no longer the bottleneck at all.

## Edge-AI tradeoffs

| Factor | Fixed duty-cycle wake | Wake-on-event |
|---|---|---|
| Always-on hardware cost | none | small continuous current draw for the trigger sensor |
| Responsiveness | bounded by cycle period (misses events between wakes) | immediate, event-driven |
| Energy at low event rates | wasteful — wakes even when nothing happens | very efficient |
| Design complexity | simple (a timer) | needs a genuinely low-power always-on front-end circuit |
| Best fit | predictable, periodic sensing needs (e.g. environmental logging) | rare, bursty events (wake word, motion, anomaly) |

## Exercise

Extend `simulate_wake_on_event` to model a **hybrid** strategy: a low
duty-cycle timer wake (say, once every 10 minutes, to catch anything the
event trigger might have missed) *combined with* event-driven wakes.
Compute the combined daily energy and lifetime, and compare it against
pure wake-on-event — this hybrid pattern is common in real deployments
specifically because a purely analog trigger can have false negatives,
and the periodic backstop wake trades a small, bounded energy cost for
protection against that failure mode.
