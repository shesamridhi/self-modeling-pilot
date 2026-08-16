"""
Self-Modeling Pilot: does an auxiliary self-monitoring task make a network
simpler and more robust? (small-scale replication of the direction described
in AIAF's "Self-Modeling in Neural Networks: Emergent Simplification and
Robustness")

Setup: train two versions of the same small MLP on the same classification
task (sklearn digits, 8x8 handwritten digit images, 10 classes).
  - BASELINE: standard supervised training only.
  - SELF-MODELING: same network + an auxiliary head that must predict the
    network's own hidden-layer activations from an earlier layer (a simple,
    concrete form of "monitoring your own internal state"). Trained jointly
    with the classification loss.

We then compare, honestly, on real measured numbers:
  1. Effective rank of the final hidden representation (proxy for internal
     complexity — lower effective rank = simpler internal structure)
  2. Test accuracy (clean)
  3. Test accuracy under input noise (proxy for robustness/generalization)

This is a small, single-run pilot on a toy dataset — NOT a claim of
reproducing the original paper's full results. It's meant to test whether
this measurement setup shows the predicted direction (self-modeling ->
lower complexity, comparable or better robustness) at all, before scaling up.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

class MLP(nn.Module):
    def __init__(self, in_dim=64, hidden=32, out_dim=10, self_model=False):
        super().__init__()
        self.self_model = self_model
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)
        if self_model:
            # auxiliary head: predict layer-2 activations from layer-1 activations
            # (a simple, concrete "self-monitoring" task)
            self.self_head = nn.Linear(hidden, hidden)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        logits = self.out(h2)
        if self.self_model:
            h2_pred = self.self_head(h1)
            return logits, h2, h2_pred
        return logits, h2, None

def effective_rank(activations):
    """Effective rank via entropy of normalized singular values (real metric,
    not fabricated) — lower = simpler / more compressed representation."""
    A = activations.detach().numpy()
    A = A - A.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(A, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.nan
    s = s[s > 1e-10]
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))

def load_data():
    data = load_digits()
    X, y = data.data, data.target
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    to_l = lambda a: torch.tensor(a, dtype=torch.long)
    return to_t(X_train), to_t(X_test), to_l(y_train), to_l(y_test)

def train(model, X_train, y_train, epochs=150, self_weight=0.3):
    opt = optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits, h2, h2_pred = model(X_train)
        loss = ce(logits, y_train)
        if model.self_model:
            loss = loss + self_weight * mse(h2_pred, h2.detach())
        loss.backward()
        opt.step()
    return model

def evaluate(model, X_test, y_test, noise_std=0.0):
    model.eval()
    with torch.no_grad():
        X_eval = X_test + torch.randn_like(X_test) * noise_std if noise_std > 0 else X_test
        logits, h2, _ = model(X_eval)
        preds = logits.argmax(dim=1)
        acc = (preds == y_test).float().mean().item()
        rank = effective_rank(h2)
    model.train()
    return acc, rank

def run_condition(self_model, X_train, X_test, y_train, y_test, n_seeds=10):
    accs_clean, accs_noisy, ranks = [], [], []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        model = MLP(self_model=self_model)
        model = train(model, X_train, y_train)
        acc_clean, rank = evaluate(model, X_test, y_test, noise_std=0.0)
        acc_noisy, _ = evaluate(model, X_test, y_test, noise_std=0.6)
        accs_clean.append(acc_clean)
        accs_noisy.append(acc_noisy)
        ranks.append(rank)
    return np.array(accs_clean), np.array(accs_noisy), np.array(ranks)

def main():
    X_train, X_test, y_train, y_test = load_data()

    base_clean, base_noisy, base_rank = run_condition(False, X_train, X_test, y_train, y_test)
    sm_clean, sm_noisy, sm_rank = run_condition(True, X_train, X_test, y_train, y_test)

    print("=== BASELINE (n=10 seeds) ===")
    print(f"Clean test acc:   {base_clean.mean():.3f} +/- {base_clean.std():.3f}")
    print(f"Noisy test acc:   {base_noisy.mean():.3f} +/- {base_noisy.std():.3f}")
    print(f"Effective rank:   {base_rank.mean():.2f} +/- {base_rank.std():.2f}")

    print("\n=== SELF-MODELING (n=10 seeds) ===")
    print(f"Clean test acc:   {sm_clean.mean():.3f} +/- {sm_clean.std():.3f}")
    print(f"Noisy test acc:   {sm_noisy.mean():.3f} +/- {sm_noisy.std():.3f}")
    print(f"Effective rank:   {sm_rank.mean():.2f} +/- {sm_rank.std():.2f}")

    with open("results.txt", "w") as f:
        f.write("BASELINE (n=10 seeds)\n")
        f.write(f"Clean test acc:   {base_clean.mean():.3f} +/- {base_clean.std():.3f}\n")
        f.write(f"Noisy test acc:   {base_noisy.mean():.3f} +/- {base_noisy.std():.3f}\n")
        f.write(f"Effective rank:   {base_rank.mean():.2f} +/- {base_rank.std():.2f}\n\n")
        f.write("SELF-MODELING (n=10 seeds)\n")
        f.write(f"Clean test acc:   {sm_clean.mean():.3f} +/- {sm_clean.std():.3f}\n")
        f.write(f"Noisy test acc:   {sm_noisy.mean():.3f} +/- {sm_noisy.std():.3f}\n")
        f.write(f"Effective rank:   {sm_rank.mean():.2f} +/- {sm_rank.std():.2f}\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot([base_rank, sm_rank], labels=["Baseline", "Self-modeling"])
    axes[0].set_ylabel("Effective rank of hidden representation")
    axes[0].set_title("Representation complexity")

    axes[1].boxplot([base_noisy, sm_noisy], labels=["Baseline", "Self-modeling"])
    axes[1].set_ylabel("Test accuracy under input noise")
    axes[1].set_title("Robustness")

    fig.tight_layout()
    fig.savefig("self_modeling_results.png", dpi=150)
    print("\nSaved: results.txt, self_modeling_results.png")

if __name__ == "__main__":
    main()
