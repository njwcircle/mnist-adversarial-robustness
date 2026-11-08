import torch
import torch.nn.functional as F
from train import CNN, get_loaders, evaluate, device


def load_model(path="mnist_cnn.pt"):
    model = CNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)

    loss = F.cross_entropy(model(x), y)
    model.zero_grad()
    loss.backward()

    x_adv = x + eps * x.grad.sign()
    return x_adv.clamp(0, 1).detach()


def pgd(model, x, y, eps, alpha=None, steps=40, random_start=True):
    if alpha is None:
        alpha = 2.5 * eps / steps
    x = x.clone().detach()  # 원본

    if random_start:
        x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    else:
        x_adv = x.clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        model.zero_grad()
        loss.backward()

        x_adv = x_adv.detach() + alpha * x_adv.grad.sign()

        delta = (x_adv - x).clamp(-eps, eps)
        x_adv = (x + delta).clamp(0, 1).detach()
    return x_adv


def eval_attack(model, loader, attack, eps):
    correct = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        X_adv = attack(model, X, y, eps)
        with torch.no_grad():
            pred = model(X_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
    return correct / len(loader.dataset)


def main():
    _, test_loader = get_loaders()
    model = load_model()

    print(f"clean accuracy: {evaluate(model, test_loader):.4f}")

    X, y = next(iter(test_loader))
    X, y = X.to(device), y.to(device)
    for name, attack in [("fgsm", fgsm), ("pgd", pgd)]:
        X_adv = attack(model, X, y, 0.1)
        print(f"{name}: max |x_adv - x| at eps=0.1: {(X_adv - X).abs().max().item():.4f}")

    print(f"{'eps':<6}{'FGSM':>8}{'PGD':>8}")
    for eps in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        acc_fgsm = eval_attack(model, test_loader, fgsm, eps)
        acc_pgd = eval_attack(model, test_loader, pgd, eps)
        print(f"{eps:<6}{acc_fgsm:>8.4f}{acc_pgd:>8.4f}")


if __name__ == "__main__":
    main()
