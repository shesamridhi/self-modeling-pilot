# Self-Modeling Pilot: Does Self-Monitoring Simplify a Network?

**Question:** AIAF's research shows that networks given a self-monitoring task
spontaneously simplify (lower internal complexity) and become more robust,
inspired by Attention Schema Theory. Does a small-scale, single-day version of
this setup show the same direction, even weakly?

## Setup

Two versions of the same small MLP, trained on `sklearn.datasets.load_digits`
(8x8 handwritten digits, 10 classes), averaged over 10 random seeds each:

- **Baseline:** standard supervised training only.
- **Self-modeling:** same network + an auxiliary head that must predict the
  network's own layer-2 activations from its layer-1 activations — a simple,
  concrete form of "monitoring your own internal state," trained jointly with
  the classification loss.

Measured, honestly, with real numbers (no cherry-picking):
1. **Effective rank** of the final hidden representation (entropy of singular
   values) — lower = simpler internal structure.
2. **Clean test accuracy.**
3. **Test accuracy under input noise** (proxy for robustness/generalization).

## Results (this run, n=10 seeds each)

```
BASELINE
Clean test acc:   0.949 +/- 0.007
Noisy test acc:   0.894 +/- 0.008
Effective rank:   16.54 +/- 0.28

SELF-MODELING
Clean test acc:   0.947 +/- 0.008
Noisy test acc:   0.893 +/- 0.008
Effective rank:   16.20 +/- 0.35
```

## Honest interpretation

The effective rank is slightly lower for the self-modeling condition
(16.20 vs 16.54), in the predicted direction, but the difference is **small
relative to the spread across seeds** — not a strong or clearly significant
effect at this scale. Accuracy and robustness are essentially unchanged
between the two conditions.

**This is a negative-to-inconclusive result at this scale, and I'm reporting
it as such rather than overstating it.** Plausible reasons the effect is weak
here:
- The network is tiny (32 hidden units) and the task (digit classification)
  is close to saturating — there may not be much "unnecessary complexity" for
  self-modeling to remove in the first place.
- The auxiliary self-modeling task used here (predict layer-2 from layer-1)
  is a simplified stand-in for what the original research likely used, and
  may be too weak a self-monitoring signal.
- Only 150 training epochs, single architecture, single dataset.

## What I'd actually test next

- Use a network with deliberately more capacity than the task needs (where
  "unnecessary complexity" has more room to exist), and check if the effect
  gets stronger.
- Try a harder/noisier version of the task where robustness differences would
  have more room to show up.
- Try a more faithful self-modeling objective — predicting a compressed
  summary of the network's *own* full internal state, not just one layer from
  another.
- Scale up hidden size and training length to see if the gap between
  conditions widens with capacity, matching the "unnecessary complexity" story.

## Files

- `self_modeling.py` — full experiment code (runnable, `python3 self_modeling.py`)
- `results.txt` — output metrics from the run above
- `self_modeling_results.png` — boxplots comparing both conditions

## Author

Samridhi Singh — B.Tech CSE (2022–2026), Jagannath University. Small-scale
pilot exploring the self-modeling/self-monitoring direction in AI alignment.
