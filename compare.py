import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import get_loaders, evaluate
from attack import load_model, fgsm, pgd, eval_attack

# standard 모델 vs adversarially trained 모델을 같은 공격으로 비교

EPS_LIST = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
MODELS = {"standard": "mnist_cnn.pt", "adv-trained": "mnist_cnn_adv.pt"}


def main(out="results/robust_accuracy.png"):
    _, test_loader = get_loaders()
    results = {}
    for name, path in MODELS.items():
        model = load_model(path)
        print(f"[{name}] clean accuracy: {evaluate(model, test_loader):.4f}", flush=True)
        results[name] = {
            "FGSM": [eval_attack(model, test_loader, fgsm, e) for e in EPS_LIST],
            "PGD": [eval_attack(model, test_loader, pgd, e) for e in EPS_LIST],
        }

    header = f"{'eps':<6}" + "".join(f"{n + ' ' + a:>20}" for n in MODELS for a in ("FGSM", "PGD"))
    print(header)
    for i, e in enumerate(EPS_LIST):
        row = f"{e:<6}" + "".join(f"{results[n][a][i]:>20.4f}" for n in MODELS for a in ("FGSM", "PGD"))
        print(row)

    fig, ax = plt.subplots(figsize=(6, 4))
    styles = {"standard": "tab:red", "adv-trained": "tab:blue"}
    for name in MODELS:
        ax.plot(EPS_LIST, results[name]["PGD"], "o-", color=styles[name], label=f"{name} (PGD)")
        ax.plot(EPS_LIST, results[name]["FGSM"], "s--", color=styles[name], alpha=0.5, label=f"{name} (FGSM)")
    ax.set_xlabel("epsilon (L-inf)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
