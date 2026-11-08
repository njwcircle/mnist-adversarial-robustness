import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from train import get_loaders, device
from attack import load_model, fgsm


def predict(model, x):
    with torch.no_grad():
        logits = model(x)
    probs = logits.softmax(dim=1)
    conf, pred = probs.max(dim=1)
    return pred, conf


def main(n=6, eps_list=(0.1, 0.2, 0.3), out="results/fgsm_examples.png"):
    _, test_loader = get_loaders()
    model = load_model()

    X, y = next(iter(test_loader))
    X, y = X.to(device), y.to(device)

    pred, _ = predict(model, X)
    idx = (pred == y).nonzero().flatten()[:n]
    X, y = X[idx], y[idx]

    cols = 1 + len(eps_list)
    fig, axes = plt.subplots(n, cols, figsize=(2.2 * cols, 2.3 * n))

    images = [("original", X)] + [(f"eps={e}", fgsm(model, X, y, e)) for e in eps_list]

    for c, (name, imgs) in enumerate(images):
        pred, conf = predict(model, imgs)
        for r in range(n):
            ax = axes[r, c]
            ax.imshow(imgs[r, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            correct = pred[r].item() == y[r].item()
            ax.set_title(f"pred {pred[r].item()} ({conf[r].item():.0%})",
                         fontsize=9, color="green" if correct else "red")
            if r == 0:
                ax.annotate(name, xy=(0.5, 1.28), xycoords="axes fraction",
                            ha="center", fontsize=11, fontweight="bold")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
